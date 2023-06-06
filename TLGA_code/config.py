import os
import torch
from numpy import random
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
root_dir = "/data1/lhz/pg_network_torch/data/weibo_finance"
train_data_path = os.path.join(root_dir, "train.bin")
eval_data_path = os.path.join(root_dir, "val.bin")
decode_data_path = os.path.join(root_dir, "test.bin")
vocab_path = os.path.join(root_dir, "vocab")
log_root = "/data2/lhz/learned_pe"
if not os.path.isdir(log_root):
    os.makedirs(log_root)

# parameters
hidden_dim = 256
emb_dim = 512
batch_size = 16
max_enc_steps = 400
max_dec_steps = 50
beam_size = 4

min_dec_steps = 16
vocab_size = 6000

lr = 0.15
adagrad_init_acc = 0.1
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4
max_grad_norm = 2.0

pointer_gen = True
#pointer_gen = False
# is_coverage = True
is_coverage = False
cov_loss_wt = 1

eps = 1e-12
max_iterations = 3000000

lr_coverage=0.15

# 使用GPU
use_gpu = True
GPU = "cuda:0"
USE_CUDA = use_gpu and torch.cuda.is_available()     # 是否使用GPU
NUM_CUDA = torch.cuda.device_count()
DEVICE = torch.device(GPU if USE_CUDA else 'cpu')

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
if USE_CUDA:
    torch.cuda.manual_seed_all(SEED)
