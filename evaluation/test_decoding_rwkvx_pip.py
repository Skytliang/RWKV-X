########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
#
# pip install rwkv lm_eval --upgrade
# previous version only support lm_eval==0.3.0
# this version support lm_eval>=0.4.0
#
import os, sys, types, json, math, time
import argparse
import random
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
from torch.nn import functional as F

os.environ["RWKV_JIT_ON"] = '0'
os.environ["RWKV_CUDA_ON"] = '1'
os.environ["RWKV_V7_ON"] = "1"
from rwkv_x.model import RWKV_X

 
########################################################################################################
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('model_path', type=str)
    parser.add_argument('--log_dir', type=str, default='logs/decoding/')
    parser.add_argument('--device', type=str, default='cuda:0')
    # add a group for moba
    group = parser.add_argument_group('moba')
    group.add_argument('--moba_chunk_size', type=int, default=2048, help='chunk size for moba')
    group.add_argument('--moba_topk', type=int, default=3, help='topk for moba')
    # add a group for eval
    group = parser.add_argument_group('eval')
    group.add_argument('--max_seq_lengths', type=int, nargs='+', default=[1000, 2000, 4000, 8000], help='max sequence lengths for ruler')

    args = parser.parse_args()
    return args

args = parse_config()
MODEL_NAME = args.model_path
OUTPUT_DIR = Path(args.log_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f'Loading model - {MODEL_NAME}')
torch.cuda.set_device(args.device)
model = RWKV_X(model_path=args.model_path, strategy='cuda fp16')
print(f"Model loaded on {args.device}")

# measure decoding latency
records = []
for ctx_len in args.max_seq_lengths:
    ctx = [0] * ctx_len
    out, state = model.forward(ctx, None)
    start_time = time.time()
    for i in range(10):
        out, state = model.forward([i], state)
    end_time = time.time()
    latency = (end_time - start_time) / 10
    print(f"ctx_len: {ctx_len}, latency: {latency * 1000:.2f} ms")

    # measure gpu memory usage
    mem_alloc = torch.cuda.memory_allocated(args.device) / 1024 ** 3 # in GiB
    print(f"ctx_len: {ctx_len}, memory: {mem_alloc:.2f} GiB")
    torch.cuda.reset_peak_memory_stats(args.device)
    torch.cuda.empty_cache()
    records.append(dict(ctx_len=ctx_len, latency=latency, memory=mem_alloc))
# first column is ctx_len, second column is latency, third column is memory
df = pd.DataFrame(records)
df.to_csv(OUTPUT_DIR / Path(args.model_path).stem() + '_decoding.csv', index=False)
print(f"Decoding latency and memory usage saved to {OUTPUT_DIR / Path(args.model_path).stem() + '_decoding.csv'}")
