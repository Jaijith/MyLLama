# mypy: ignore-errors
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing_extensions import Self

from domi.utils import find_multiple, domi_config

"""
This cache stores the mask used in the attention computation as they 
dont change throughout the sequance generation process.

Its a 4D tensor. So if we take a specific value from this 4D tensor, 
say mask_cache[a][b][c][d], it represents whether the c-th token in a 
sequence of length b can attend to the d-th token. 
If the value is 1, it means attention is disallowed from token c to 
token d; if it's 0, attention is allowed.

In the context of the sentence "India is my country", 
with a block size of 5, the tensor would be of shape [5, 5, 5, 5]. 
For instance, mask_cache[4][3][1][4] would tell us if the second token 
in a sequence of length 4 can attend to the fifth token. 
Since our sentence only has 4 tokens, the mask value would be 1, 
indicating that attention is disallowed because the fifth token 
doesn't exist in our sentence.

"""
MaskCache = torch.Tensor

"""
This cache Store the Rotary Position Embedding avoiding comutiing them
again and again as they dont chage over differnt runs. 

The size of rope_cache is typically [block_size, n_embd], where 
block_size is the length of the block (i.e., sequence length) and 
n_embd is the dimensionality of the embeddings.
"""
RoPECache = torch.Tensor

"""
This cache stores the key and value tensors in the self-attnetion mechanis of the transformer
the key and value of each postion up to the current positon remais the same.

Each tensor is a 4D tensor of size [B, n_head, max_seq_length, 
head_size], where B is the batch size, n_head is the number of attention 
heads, max_seq_length is the maximum sequence length, and head_size 
is the size of each head (which is n_embd divided by n_head).
"""
KVCache = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class DomiConfig:
    """
    block_size:  this is the max length of input sequence
    
    vocab_size: this is the size of the vocab the model is using. This
    directly affects the size of the embeddings layer, as each token in
    the vocab needs an embedding
    
    padded_vocab_size: this param is used for tensor operations that
    may need the vocab size to be a multiple of a certain number. If its
    not provided, its computed using the 'find_multiple' function to be 
    the smallest multiple of 64 that is greater than or equal to vocab_size
    
    n_layer: the number of layers in the transformer model, here a 
    transfomer blocks that includes self attention and feed forward NN
    
    n_head: the number of attention heads in the self-attention mechanism.
    each head allows the model to focus on different part of th input when
    computing attnetion output
    
    n_embd: dimensionality of the embeddings in the model. this is also the 
    size of the hidden state in the transfomation layers
    
    __post_init__: special python method thats called afger the class is
    initialized. here it is used to compute the 'padded_vocab_size'
    
    from_name ->self: allows to create an isnatnce of DomiConfig by
    providing the name of a preset congifuation., in our case domi_configs
    defined later where keys are config names and values are dict of params.
    The mthod will fetch the params for specified config and use them to create
    an instance of the DomiConfig
    
    """
    block_size: int = 2048
    vocab_size: int = 32000
    padded_vocab_size: Optional[int] = None
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096

    def __post_init__(self):
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(self.vocab_size, 64)

    @classmethod
    def from_name(cls, name: str) -> Self:
        return cls(**domi_config[name])


class DOMI(nn.Module):
    def __init__(self, config: DomiConfig) -> None:
        
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config
        """
         lm_head maps from the dimension of the model's internal representation 
         (config.n_embd) to the size of the vocabulary (config.padded_vocab_size), 
         allowing the model to produce a probability distribution 
         over all possible words in the vocabulary for its output.
        """
        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=RMSNorm(config.n_embd),
            )
        )

        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[MaskCache] = None
        self.kv_caches: List[KVCache] = []

    def _init_weights(self, module: nn.Module) -> None:
        """
        This is a method to initialize the weights of the model's layers. 
        It uses different initialization depending on whether the 
        module is a linear layer or an embedding layer. It uses the 
        normal distribution to initialize weights.
        """

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))
            

    def forward(
        self, idx: torch.Tensor, max_seq_length: Optional[int] = None, input_pos: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[KVCache]]]:
        """
        Let's assume we're dealing with a batch of size B, each consisting of a sequence of length T, 
        and we're using the configuration "7B" as the LLaMAConfig for this model. So, according to the "7B" 
        configuration, n_embd=4096, n_layer=32, n_head=32, vocab_size=32000, padded_vocab_size will be the 
        smallest multiple of 64 greater than or equal to vocab_size, and block_size=2048.
        
        Input: The forward pass receives an input tensor idx with shape (B, T), 
        where B is the batch size and T is the sequence length. For instance, let's say our batch size 
        is 10 and sequence length is 2000, so the input tensor size is (10, 2000).
        
        Initial Checks: The forward function first verifies the lengths of the sequences, block size, 
        and maximum sequence length. If these values are not set, they are defaulted to the block_size 
        from the LLaMAConfig which is 2048.
        
        Then the forward model goes ahead and builds the cache for rope and mask if they dont exist.
        
        IF input_pos is not None, it used the Pytorch function 'index_select' to seelct positional encodings 
        and masks correspoding to the input position. 'index_slect' takes an input tensor, a dimension along
        which to index(0 for rope and 2 for masks) and the indices to select. In this case, input_pos is a
        tensor of indices , so it selects the position encodings and masks corresponding to these positions.
        
        
        
        """
        B, T = idx.size()

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        assert T <= max_seq_length, f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        assert max_seq_length <= block_size, f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert T <= block_size, f"Cannot forward sequence of length {T}, block size is only {block_size}"

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx)
        if self.mask_cache is None:
            self.mask_cache = self.build_mask_cache(idx)

        if input_pos is not None:
            # Selects the postional encoding that corresponds to the positions in 'input_pos'
            rope = self.rope_cache.index_select(0, input_pos)
            # Selects the attention masks corresponding the the postions in 'input_pos'
            mask = self.mask_cache.index_select(2, input_pos)
            # Trims the mask to the maximum sequence length.
            mask = mask[:, :, :, :max_seq_length]
        else:
            rope = self.rope_cache[:T]
            mask = self.mask_cache[:, :, :T, :T]

        # forward the model itself
        """
        The input tensor 'idx'(batch_size, sequence_lenght) is passed on to the embedding layer.
        it results in a 3-dim tensor 'x' of shape (b, t, n_embed)
        """
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        if input_pos is None:  # proxy for use_cache=False
            """
            processes the sequence without caching
            """
            for block in self.transformer.h:
                x, _ = block(x, rope, mask, max_seq_length)
        else:
            if not self.kv_caches:
                """
                if cache is not initialized, it will be created with the shape.
                The cache is a list of tuples (key and value) of zeros tensors, with each tuple
                corresponding to a layer in the transformation model.
                
                then the input tensor is again passed through each transformation block, but this time
                the block also receives input position and the corresponding cache. the key value pair
                caches returned by the blocks are saved back into kv_cache.
                """
                head_size = self.config.n_embd // self.config.n_head
                cache_shape = (B, self.config.n_head, max_seq_length, head_size)
                self.kv_caches = [
                    (torch.zeros(cache_shape, device=x.device, dtype=x.dtype), torch.zeros(cache_shape, device=x.device, dtype=x.dtype))
                    for _ in range(self.config.n_layer)
                ]
            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i] = block(x, rope, mask, max_seq_length, input_pos, self.kv_caches[i])


        #The output from the transfomer blocks is then normalized using the final layer normalization.        
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)  # (b, t, vocab_size)

        return logits
    
    @classmethod
    def from_name(cls, name: str) -> Self:
        """
        Getting the confinguration dict to use from the name, e.g. 7B
        """
        return cls(DomiConfig.from_name(name))

    def build_rope_cache(self, idx: torch.Tensor) -> RoPECache:
        """
        The function calls the below build rope cache function and
        builds the cache for RoPE
        """
        return build_rope_cache(
            seq_len=self.config.block_size,
            n_elem=self.config.n_embd // self.config.n_head,
            dtype=idx.dtype,
            device=idx.device,
        )

    def build_mask_cache(self, idx: torch.Tensor) -> MaskCache:
        """
        This function builds a cache for the attenstion masks used in the transformer
        model. This is used to prevent attention from further position in the
        sequence. 
        
        A metric of ones with dimension (block_size, block_size) is created.
        The upper traingular part of the matrix is set to zero using the
        'torch.tril()' function which means only the lower triangle and diagonals
        are filled with ones
        The resulting mask is unsqueezed twice to add two dimensions at the front, 
        resulting in 4D tensor
        """
        ones = torch.ones((self.config.block_size, self.config.block_size), device=idx.device, dtype=torch.bool)
        return torch.tril(ones).unsqueeze(0).unsqueeze(0)

    def reset_cache(self) -> None:
        """
        Used to reset the cache.
        it also has a specific condition when cache is stored in 'xla' device type
        in which case the rope and mask caches are set to None. XLA refers to
        google accelerated linear algebra, a domina sepcific compiler for linear
        algebra that can be used to acclearate TensorFlow computations.
        """
        self.kv_caches.clear()
        if self.mask_cache.device.type == "xla":
            # https://github.com/Lightning-AI/lit-parrot/pull/83#issuecomment-1558150179
            self.rope_cache = None
            self.mask_cache = None
    
    
class CausalSelfAttention(nn.Module):
    """
    In the constructor two linear layers c_attn and c_proj are intialized
    The c_attn layer transforms the input to create queries, keys and values
    for each attention head, while the 'c_proj' layer transforms the output
    of the attention mechanism back to its original dimension.
    
    In the forward layer, the input x is first transformed by c_attn into
    queries, keys and values. Then apply_rote method applies the rotary position
    embedding  to both queries and keys. 
    
    Then the dimensions of Q,K,V are permuted so that the head dimension comes
    before the sequence length. This is done to facilitate the subsequent  matrix 
    operations for calculating the subsequent matrix opeartions for calculating
    the attention score
    
    If the cache is not 'None' we are using cached results to improve computational
    efficiency, typically when genrating sequences token by token (like in autoregressive
    decoding). In this case, the caceh is updated to store the current keys and values.
    
    The scaled dot porduct attnetion computed the dot product of the queries and the keys,
    scales them, applied a mask, and then applies a softmax function to obtain the final
    attention score. These scores are used to create a weighted sum of the values 'v'
    yielding the output of the attention mechanism. 'y
    
    The output tensor y is reshaped to combine the results from differnt attention heads
    side by side, and then it is transformed back to the original embedding dimension by
    the 'c_proj' layer.
    """
    def __init__(self, config: DomiConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        mask: MaskCache,
        max_seq_length: int,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size)
        q = q.view(B, T, self.n_head, head_size)
        v = v.view(B, T, self.n_head, head_size)

        q = apply_rope(q, rope)
        k = apply_rope(k, rope)

        k = k.transpose(1, 2)  # (B, nh, T, hs)
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)

        if kv_cache is not None:
            #Unpacking the cache keys and values from the previous steps. 
            # size is (batch_size, num_heads, seq_lenght, head_size)
            cache_k, cache_v = kv_cache
            # check if reached token limit. if true, cache has reached its limit
            #we need to make room for new tokens
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # shift all enties in the cache 1 position to the left.
                #the oldest key value pair is discarded and a new pair is opened up at the end
                cache_k = torch.roll(cache_k, -1, dims=2)
                cache_v = torch.roll(cache_v, -1, dims=2)
            #Update the cache at the position given by input_pos with the newly computed 
            # keys 'k' & values 'v
            k = cache_k.index_copy(2, input_pos, k)
            v = cache_v.index_copy(2, input_pos, v)
            kv_cache = k, v

        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)

        return y, kv_cache
    

class Block(nn.Module):
    """
    in the forward layer, first RMS normalization is applied to the
    input 'x', then passes the normalized data through the self -attention
    mechanism. 'rope' and 'mask' are positional encoding ans mask resp,
    and kv_cache is a chace for the key value pairs in the self-attention
    mechanism. The output 'h' is the result of applying the self-attention 
    and new_kv_cache is the updated cache of key value pair
    
    x = x + h . This line adds the output of self attention layer to the
    original input. This is so called residual connection which helps
    mitigate the vanishing graidient problem in deep networks
    
    x = x + self.mlp(self.rms_2(x)). The input is again passed through the
    RMS normalization , then passed to MLP . The result is added to the
    input for another residual connection.
    
    The forward layer returns the final 'x' and the updates key-value
    cache 'new_kv_cache'
    
    This blocks contatins a single transformation block. There will be
    multiple such blocks stacked one on top of each other.
    """
    def __init__(self, config: DomiConfig) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.rms_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        mask: MaskCache,
        max_seq_length: int,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        h, new_kv_cache = self.attn(self.rms_1(x), rope, mask, max_seq_length, input_pos, kv_cache)
        x = x + h
        x = x + self.mlp(self.rms_2(x))
        return x, new_kv_cache

class MLP(nn.Module):
    """
    In the constructor, the hidden_dim is kept as 4 times the embed dim.
    and n_hidden as 2/3 of hidden_dim. The find mulitple adjust n_hidden
    so that it is a multiple of 256 which can make computations more
    efficient.
    
    The first c_fc1 is a linear transformation with input dim config.n_embed 
    and output og n_hidden. The second layer c_fc2 is also a linear transformation
    with same dimension. This c_proj projects from n_hidden state back to
    embedding dimension.
    
    In the forwarding method, input x is first transformed by c_fc1 and then
    an activation function sigmoid linear unit is applied. this is element
    wise multiplied by c_fc2 layer. This is most likely for residual connection
    
    The resul is then transformed by c_proj and this final transformed version of
    x is returned.
    """
    
    def __init__(self, config: DomiConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd
        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, 256)

        self.c_fc1 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x

class RMSNorm(nn.Module):
    """
    RMS normalization operates on the given tensor x. It is similar to
    standard layer normalization but instead of using the mean and variance,
    it uses the square root of the mean of the square elements. This provides
    a scaling factor for the normalization which can be beneficial.
    
    The constructor initialized a sacle parameter as a learnable pytorch
    parameter and sets the epsilon(used for numerical stability) and the 
    dimension on which the normalization should be performed.
    
    The forward method calculates the root mean suqare of x.
    Then it calculates x_normed by scaling x with the reciprocal square 
    root of the norm.
    The scaled tensor is then multiplied by the learnable 'scale' parameter
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed


def build_rope_cache(
    seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000
) -> RoPECache:
    """
    This function generates the rotational embedding matrix, which is the same size as the input sequence 
    length seq_len and dimensionality n_elem. base is the constant used in calculating the rotation angles.
    
    The purpose of this function is to prepare a cache of rotation angles, theta, that are multiplied 
    with the position index, seq_idx, to generate idx_theta. Finally, the function returns a cache that 
    contains the cosine and sine values of idx_theta. The dtype and device adjustments are made to 
    ensure the calculations are consistent with the original implementation.
    
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).float()

    cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        cache = cache.half()
    return cache


def apply_rope(x: torch.Tensor, rope_cache: RoPECache) -> torch.Tensor:
    """
    This function applies the rotational positional embedding to the input tensor x. 
    It first adjusts the rope_cache size to match the input sequence length. 
    Then, the input tensor x and rope_cache are reshaped to facilitate element-wise multiplication and 
    addition operations. These operations are then applied, essentially rotating the original embeddings 
    in the complex plane. The final tensor x_out2 is reshaped and returned with the same dtype as the 
    input tensor x.
    
    Now let's create an example:

    Suppose we have an input tensor x of size (batch_size=2, seq_len=5, n_elem=6). 
    Here, we have 2 sequences, each of length 5 and each element in the sequence is represented 
    by an embedding of size 6.
    
    After running this code, it will be a tensor of the same size as x but with the positional information 
    encoded through rotations.

    Regarding the size of rope_cache, it should be a tensor of shape (seq_len, n_elem // 2, 2), 
    which comes from stacking cosine and sine values of the product of the sequence index and theta. 
    In our example, rope_cache would have a shape of (5, 3, 2).
    """
    # truncate to support variable sizes
    T = x.size(1)
    rope_cache = rope_cache[:T]

    # cast because the reference does
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)

