#For Training

#Number of chunks to use for training
n_chunks = 4

#Number of devices to train on
n_device = 4

#Traning model to use

model_type = "7B"

# Location of the training director
train_data_dir = "data/train/"

# Hyderparameters
learning_rate = 6e-4
batch_size = 125
micro_batch_size = 5
max_iters = 600000 # num_epochs * (epoch_size // micro_batch_size) // devices
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = max_iters
min_lr = 6e-5

# Intervals 
out_dir = "out/training"
save_interval = 1000
eval_interval = 1000
eval_iters = 100
log_interval = 1



