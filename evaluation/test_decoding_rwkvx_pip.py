########################################################################################################
# The RWKV-X Language Model - https://github.com/howard-hou/RWKV-X
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
from rwkv_x.model import RWKV_X, RWKV_X_Config

 
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
    group.add_argument('--attn_mode', type=str, default='sparse', choices=['full', 'sparse'], help='attention mode')
    # add a group for kv cache
    group = parser.add_argument_group('kv_cache')
    group.add_argument('--max_kv_cache_size', type=int, default=0, help='0 means no kv cache management')
    group.add_argument('--kv_cache_window_size', type=int, default=2000, help='0 means no kv cache management')
    group.add_argument('--min_kv_cache_size', type=int, default=16000, help='kv cache size keep after management')
    # add a group for eval
    group = parser.add_argument_group('eval')
    group.add_argument('--max_seq_lengths', type=int, nargs='+', default=[1000, 2000, 4000, 8000], help='max sequence lengths for ruler')

    args = parser.parse_args()
    return args

args = parse_config()
MODEL_NAME = args.model_path
MODEL_STEM = Path(MODEL_NAME).stem
OUTPUT_DIR = Path(args.log_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f'Loading model - {MODEL_NAME}')
torch.cuda.set_device(args.device)
config = RWKV_X_Config(
        moba_chunk_size=args.moba_chunk_size,
        moba_topk=args.moba_topk,
        attn_mode=args.attn_mode,
        max_kv_cache_size=args.max_kv_cache_size,
        kv_cache_window_size=args.kv_cache_window_size,
        min_kv_cache_size=args.min_kv_cache_size,
    )
print('Model Config:', config)
model = RWKV_X(model_path=args.model_path, strategy='cuda fp16', config=config)
print(f"Model loaded on {args.device}")
print(f"Context length Test: {args.max_seq_lengths}")

# measure prefill and decoding latency & memory
prefill_records = []
decoding_records = []
chunk_len = 4000

for ctx_len in args.max_seq_lengths:
    tokens = [0] * ctx_len

    # Measure prefill
    torch.cuda.reset_peak_memory_stats(args.device)
    torch.cuda.empty_cache()
    state = None
    start_prefill = time.time()
    while len(tokens) > 0:
        out, state = model.forward(tokens[:chunk_len], state)
        tokens = tokens[chunk_len:]
    end_prefill = time.time()
    prefill_latency = (end_prefill - start_prefill) * 1000  # in ms
    prefill_mem = torch.cuda.max_memory_allocated(args.device) / 1024 ** 3  # in GiB
    print(f"[PREFILL] ctx_len: {ctx_len}, latency: {prefill_latency:.2f} ms, memory: {prefill_mem:.2f} GiB")

    prefill_records.append(dict(ctx_len=ctx_len, latency=prefill_latency, memory=round(prefill_mem, 2)))

    # Measure decoding
    torch.cuda.reset_peak_memory_stats(args.device)
    torch.cuda.empty_cache()
    start_decode = time.time()
    for i in range(10):
        out, state = model.forward([i], state)
    end_decode = time.time()
    decoding_latency = (end_decode - start_decode) / 10 * 1000  # average per token, in ms
    decoding_mem = torch.cuda.max_memory_allocated(args.device) / 1024 ** 3  # in GiB
    print(f"[DECODING] ctx_len: {ctx_len}, latency: {decoding_latency:.2f} ms, memory: {decoding_mem:.2f} GiB")

    decoding_records.append(dict(ctx_len=ctx_len, latency=decoding_latency, memory=round(decoding_mem, 2)))

# Save results
prefill_df = pd.DataFrame(prefill_records)
decoding_df = pd.DataFrame(decoding_records)
# 合并 prefill 和 decoding 的结果
combined_df = prefill_df.merge(decoding_df, on="ctx_len", suffixes=("_prefill", "_decoding"))

# 保存合并后的结果
attn_mode = 'sparse_attention' if args.attn_mode == 'sparse' else 'full_attention'
kv_cache_config = f'with_kv_cache_min{args.min_kv_cache_size//1000}K-max{args.max_kv_cache_size//1000}K-window{args.kv_cache_window_size//1000}K'
kv_cache_mode = 'without_kv_cache_management' if args.max_kv_cache_size == 0 else kv_cache_config
combined_output = f"{MODEL_STEM}_{attn_mode}_{kv_cache_mode}.csv"
combined_df.to_csv(OUTPUT_DIR / combined_output, index=False)