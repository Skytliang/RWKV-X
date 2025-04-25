########################################################################################################
# The RWKV-X Language Model - https://github.com/howard-hou/RWKV-X
########################################################################################################

import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from .binidx import MMapIndexedDataset
from .utils import MaybeIsPrime


class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args

        self.vocab_size = args.vocab_size
        rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")

        self.data = MMapIndexedDataset(args.data_file)
        self.data_size = len(self.data._bin_buffer) // self.data._index._dtype_size
        rank_zero_info(f"Data has {self.data_size} tokens.")
        self.samples_per_epoch = self.args.epoch_steps * self.args.real_bsz

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size

        ctx_len = args.ctx_len
        req_len = ctx_len + 1
        data = self.data
        magic_prime = args.magic_prime
        step = epoch * self.samples_per_epoch + (idx * world_size) + rank
        # use a magic prime to sample the dataset deterministically yet randomly enough
        i = ((step * step * step) % magic_prime) * ctx_len

        dix = data.get(idx=0, offset=i, length=req_len).astype(int)

        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
