"""
Let's assume we have 3 binary files, each containing 1000 np.int64 elements, and we create a PackedDataset as follows:

    dataset = PackedDataset(['file1.bin', 'file2.bin', 'file3.bin'], n_chunks=2, block_size=200)

Here's what happens when we start iterating over the dataset:

Initialization: When the PackedDatasetIterator is created, it initializes its state. 
It sets up the random number generator if shuffling is enabled, and initializes the list of memory-mapped files and buffers.

Loading Chunks: The _load_n_chunks method is called, which loads n_chunks (2 in this case) chunks of data into memory. 
It does this by memory-mapping the files and creating a buffer for each file. In our example, it will load file1.bin and file2.bin into memory.

Creating Block Indexes: The _load_n_chunks method also creates a list of block indexes. 
This list determines the order in which blocks of data will be yielded by the iterator. 
If shuffling is enabled, this list is a random permutation; otherwise, it's just a range of integers. 
In our example, each file contains 1000/200 = 5 blocks, so the list of block indexes will contain 2*5 = 10 elements.

Yielding Blocks: When we start iterating over the dataset, the __next__ method of the PackedDatasetIterator is called. 
This method uses the current block index to select a block of data from the buffers, and yields this block. 
In our example, each block will contain 200 np.int64 elements.

Loading More Chunks: When all blocks from the currently loaded chunks have been yielded, the _load_n_chunks method is called again to load more chunks into memory. 
In our example, this will happen after 10 blocks have been yielded, at which point file3.bin will be loaded into memory.

Wrapping Around: If the wrap parameter is True, then when all files have been processed, the iterator will start again from the first file. 
In our example, after file3.bin has been processed, the iterator will load file1.bin and file2.bin into memory again.

This process continues until the iterator is exhausted, which happens when all blocks have been yielded and wrap is False.
"""

import os
import struct
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from data.data_config import dtypes, HDR_MAGIC, HDR_SIZE, code


class PackedDataset(IterableDataset):
    """
    A PyTorch IterableDataset for reading data from multiple files in chunks.

    Args:
        filenames (list of str): The list of file paths to read data from.
        n_chunks (int): The number of chunks to read at a time from each file.
        block_size (int): The size of each block to read from a chunk.
        seed (int, optional): The seed for the random number generator used for shuffling. Defaults to 12345.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        wrap (bool, optional): Whether to wrap around to the start of the files when the end is reached. Defaults to False.
        num_processes (int, optional): The number of processes used for data loading. Defaults to 1.
        process_rank (int, optional): The rank of the current process used for data loading. Defaults to 0.
    """
    def __init__(self, filenames, n_chunks, block_size, seed=12345, shuffle=True, wrap=False, num_processes=1, process_rank=0):
        self._filenames = filenames
        self._n_chunks = n_chunks
        self._block_size = block_size
        self._seed = seed
        self._shuffle = shuffle
        self._wrap = wrap
        self._num_processes = num_processes
        self._process_rank = process_rank

    def __iter__(self):
        """
        Here's a breakdown of what's happening:

        self._filenames is the list of all filenames that the dataset is supposed to process.

        shard_id is the unique identifier for the current worker. If there are multiple workers, each one will have a different shard_id.

        max_num_files is the maximum number of files that can be evenly divided among all workers. 
        It's calculated as len(self._filenames) // num_shards * num_shards, 
        which rounds down the total number of files to the nearest number that can be evenly divided by the number of shards.

        num_shards is the total number of shards, calculated as the number of workers times the number of processes.

        The line filenames = self._filenames[shard_id : max_num_files : num_shards] uses Python's slice notation 
        to select every num_shards-th file from the list of filenames, starting from shard_id. 
        This effectively divides the list of filenames into num_shards equally-sized chunks, 
        and selects the chunk corresponding to shard_id.

        For example, if there are 4 shards and 12 files, this line will divide the list of filenames into
        4 chunks of 3 files each, and each worker will process one of these chunks. 
        If shard_id is 0, the worker will process the first file in each chunk; if shard_id is 1, 
        the worker will process the second file in each chunk, and so on.

        """
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        num_shards = num_workers * self._num_processes
        shard_id = self._process_rank * num_workers + worker_id
        max_num_files = len(self._filenames) // num_shards * num_shards
        filenames = self._filenames[shard_id : max_num_files : num_shards]
        return PackedDatasetIterator(
            filenames=filenames,
            n_chunks=self._n_chunks,
            block_size=self._block_size,
            seed=self._seed,
            shuffle=self._shuffle,
            wrap=self._wrap,
        )



class PackedDatasetIterator:
    """
    An iterator for reading blocks of data from the files.

    Attributes:
        _filenames: The list of filenames to read from.
        _n_chunks: The number of chunks to read at a time.
        _block_size: The number of elements in each block yielded by the iterator.
        _seed: The seed for the random number generator used to shuffle the data.
        _shuffle: Whether to shuffle the data.
        _wrap: Whether to wrap around to the beginning of the data when the end is reached.
        _rng: The random number generator used to shuffle the data.
        _file_idx: The index of the current file in the list of filenames.
        _dtype: The data type of the chunks.
        _n_blocks: The number of blocks 
    """
    def __init__(self, filenames, n_chunks, block_size, seed, shuffle, wrap):
        self._seed = seed
        self._shuffle = shuffle
        self._rng = np.random.default_rng(seed) if shuffle else None
        self._block_idxs = None

        self._wrap = wrap
        self._filenames = filenames
        self._file_idx = 0

        self._n_chunks = n_chunks

        self._dtype = None
        self._block_size = block_size
        self._n_blocks = None

        self._mmaps = []
        self._buffers = []

        self._block_idxs = []
        self._curr_idx = 0

        self._load_n_chunks()

    def _read_header(self, path):
        """
        For example, let's say we have a file data.bin in the custom binary format. 
        The file starts with a 24-byte header, followed by data chunks. 
        The header contains the magic string LITPKDS, a version number 1, 
        a data type code 5 (indicating np.int64), and the size of each chunk 1000.

        When we call _read_header('data.bin'), it will read the header of data.bin, 
        check that the magic string and version number are correct, 
        look up the data type corresponding to code 5 (np.int64), 
        and read the chunk size (1000). 
        It will then return np.int64 and 1000.
        """
        
        with open(path, "rb") as f:
            magic = f.read(len(HDR_MAGIC))
            assert magic == HDR_MAGIC, "File doesn't match expected format."
            version = struct.unpack("<Q", f.read(8))
            assert (1,) == version
            (dtype_code,) = struct.unpack("<B", f.read(1))
            dtype = dtypes[dtype_code]
            (chunk_size,) = struct.unpack("<Q", f.read(8))
        return dtype, chunk_size

    def _close_mmaps(self):
        """
        The function iterates over the list of memory-mapped files stored in self._mmaps.
        For each memory-mapped file, it calls the _mmap.close() method to close the file.
        After all the files have been closed, the function returns.
        
        This function is called in the _load_n_chunks method before new chunks are 
        loaded into memory, to ensure that the previously loaded chunks are properly cleaned up. 
        It's also called in the __del__ method, which is called when the iterator is 
        being destroyed, to ensure that all resources are freed.

        In Python, it's important to close files when you're done with 
        them to free up system resources. 
        """     
        for mmap in self._mmaps:
            mmap._mmap.close()

    def _load_n_chunks(self):
        
        """
        Closing Previous Memory-Mapped Files: The method starts by calling the 
        _close_mmaps function to close any previously opened memory-mapped files. 
        This is done to free up system resources before new files are opened.

        Checking for End of Data: The method then checks if there are enough files 
        left to load n_chunks chunks of data. If not, and if the wrap parameter is 
        False, it raises a StopIteration exception to signal the end of the data. 
        If the wrap parameter is True, it resets the file index to 0 to start again 
        from the first file.

        Loading Chunks: The method then enters a loop that runs n_chunks times. 
        In each iteration, it opens a file, reads the header to get the data type 
        and chunk size, and creates a memory-mapped file and a buffer for the file. 
        The memory-mapped file and buffer are added to the lists self._mmaps and 
        self._buffers, respectively.

        Creating Block Indexes: After all chunks have been loaded, the method creates 
        a list of block indexes. This list determines the order in which blocks of 
        data will be yielded by the iterator. If the shuffle parameter is True, 
        this list is a random permutation; otherwise, it's just a range of integers.

        Resetting Current Index: Finally, the method resets the current index to 0. 
        This index is used to keep track of which block of data should be yielded 
        next.

        In summary, the _load_n_chunks method is responsible for loading a certain 
        number of chunks of data into memory and preparing them to be yielded by 
        the iterator. It's called at the start of the iteration and whenever all 
        currently loaded blocks of data have been yielded.
        """
        
        """
        ON HOW MEMORY MAPPING WORKS
        
        When the _load_n_chunks method loads chunks of data into memory, it creates 
        a memory-mapped file for each chunk and adds it to self._mmaps. 
        It then creates a buffer for each memory-mapped file and adds it to self._buffers.

        When the __next__ method yields a block of data, it uses the current block 
        index to select a buffer from self._buffers, and then selects a slice of 
        data from this buffer. The data is then converted to a numpy array and returned.

        For example, let's say we have two binary files, file1.bin and file2.bin, 
        each containing 1000 np.int64 elements. We create a PackedDataset with 
        n_chunks=2 and block_size=200, and start iterating over it. 
        The _load_n_chunks method will load file1.bin and file2.bin into memory, 
        creating two memory-mapped files and two buffers. 
        When the __next__ method is called, it will select a slice of 200 elements 
        from one of the buffers, convert it to a numpy array, and return it.
        """
        self._close_mmaps()
        self._mmaps = []
        self._buffers = []

        if self._n_chunks > len(self._filenames[self._file_idx:]):
            if not self._wrap:
                raise StopIteration
            else:
                self._file_idx = 0

        for i in range(self._n_chunks):
            filename = self._filenames[self._file_idx + i]
            if self._dtype is None:
                self._dtype, self._chunk_size = self._read_header(
                    filename
                )
                self._n_blocks = self._chunk_size // self._block_size
            # TODO: check header matches with previous files
            mmap = np.memmap(filename, mode="r", order="C", offset=HDR_SIZE)
            self._mmaps.append(mmap)
            self._buffers.append(memoryview(mmap))

        self._file_idx += self._n_chunks
        n_all_blocks = self._n_chunks * self._n_blocks

        self._block_idxs = (
            self._rng.permutation(n_all_blocks)
            if self._shuffle
            else range(n_all_blocks)
        )
        self._curr_idx = 0
        
    def __del__(self):
        """
        _close_mmaps is called in the __del__ method, which is called when the iterator is 
        being destroyed, to ensure that all resources are freed.
        """
        self._close_mmaps()
        del self._mmaps
        del self._buffers

    def __iter__(self):
        """In Python, an iterable is an object that can be looped over, like a list or a string. 
        In order for an object to be iterable, it needs to define an __iter__() method. 
        This method should return an iterator, which is an object with a __next__() method that returns the 
        next value in the sequence each time it's called, and raises a StopIteration exception when there 
        are no more values.

        In the case of the PackedDatasetIterator class, the __iter__() method simply returns self. 
        This is because the PackedDatasetIterator class itself is an iterator. 
        It has a __next__() method that yields blocks of data from the files, and raises a StopIteration 
        exception when all blocks have been yielded and the wrap parameter is False.
        """
        
        return self

    def __next__(self):
        """
        Checking for End of Blocks: The method first checks if all blocks from the currently loaded chunks 
        have been yielded. If so, it calls the _load_n_chunks method to load more chunks into memory.

        Selecting a Block: The method then uses the current block index to select a block of data. It 
        calculates the chunk ID and the element ID within the chunk, and uses these to select a slice of 
        data from the appropriate buffer.

        Creating a Numpy Array: The method creates a numpy array from the selected slice of data. It specifies 
        the data type and the number of elements in the array, and uses the offset to start from the correct 
        position in the buffer.

        Converting to PyTorch Tensor: The method then converts the numpy array to a PyTorch tensor. 
        This is done by calling torch.from_numpy and specifying the desired data type (np.int64).

        Returning the Block: Finally, the method returns the PyTorch tensor. This is the block of data that 
        is yielded each time through the loop when you iterate over the PackedDataset.

        In summary, the __next__ method is responsible for yielding blocks of data from the files. 
        It does this by selecting slices of data from the buffers, converting them to PyTorch tensors, 
        and returning them. When all blocks from the currently loaded chunks have been yielded, 
        it loads more chunks into memory.
        """
        
        if self._curr_idx >= len(self._block_idxs):
            self._load_n_chunks()
            # TODO: trigger fetching next next n_chunks if remote
        block_idx = self._block_idxs[self._curr_idx]
        chunk_id = block_idx // self._n_blocks
        buffer = self._buffers[chunk_id]
        elem_id = (block_idx % self._n_blocks) * self._block_size
        offset = np.dtype(self._dtype).itemsize * elem_id
        arr = np.frombuffer(
            buffer, dtype=self._dtype, count=self._block_size, offset=offset
        )
        self._curr_idx += 1
        return torch.from_numpy(arr.astype(np.int64))
    

class CombinedDataset(IterableDataset):
    """
    datasets: A list of dataset you want to combine. Each dataset must
    be iterable.
    
    seed: for reproducible results
    
    weights: a list of probabilities associated with each entry in
    datasets.If not provided , it assusmes equal probability
    """
    def __init__(self, datasets, seed, weights=None):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets

    def __iter__(self):
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)


class CombinedDatasetIterator:
    """
    __init__: it converts all datasets to iterators, initializer thr random
    number generator with provided seed, and saves the weights
    
    __next__: This method is called to get the enxt element from the iterator
    It first selects on fo the datasets based on the provided weights.
    """
    def __init__(self, datasets, seed, weights):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)

    def __next__(self):
        dataset, = self._rng.choices(self._datasets, weights=self._weights, k=1)
        return next(dataset)

    
