import torch
from dataclasses import dataclass
from typing import Tuple

@dataclass
class RWKVConfig:
    n_layer: int
    n_embd: int
    n_head: int
    dim_att: int
    vocab_size: int
    head_size_a: int = 64
    head_size_divisor: int = 8
    dropout: float = 0
    enable_rwkv_ablation: bool = False
    grad_cp: int = 0

@dataclass
class MOBAConfig:
    n_moba_layer: int
    n_head: int
    n_embd: int
    moba_chunk_size: int = 2048
    moba_topk: int = 3

def load_configs_from_ckpt(path: str) -> Tuple[RWKVConfig, MOBAConfig]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    rwkv = {k[5:]: v for k, v in ckpt.items() if k.startswith("rwkv.")}
    moba = {k[5:]: v for k, v in ckpt.items() if k.startswith("moba.")}

    n_layer = len({k.split('.')[1] for k in rwkv if k.startswith("blocks.")})
    n_embd = rwkv['emb.weight'].shape[1]
    vocab_size = rwkv['emb.weight'].shape[0]
    n_head = n_embd // 64
    n_moba_layer = len({k.split('.')[0] for k in moba if k[0].isdigit()})

    return (
        RWKVConfig(n_layer=n_layer, n_embd=n_embd, dim_att=n_embd, n_head=n_head, vocab_size=vocab_size),
        MOBAConfig(n_moba_layer=n_moba_layer, n_head=n_head, n_embd=n_embd)
    )
