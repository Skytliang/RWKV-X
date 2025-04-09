import torch
from dataclasses import dataclass
from typing import Tuple

@dataclass
class MOBAConfig:
    n_moba_layer: int
    n_head: int
    n_embd: int
    moba_chunk_size: int = 2048
    moba_topk: int = 3

def load_configs_from_ckpt(path: str, moba_chunk_size=2048, moba_topk=3) -> MOBAConfig:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    rwkv = {k[5:]: v for k, v in ckpt.items() if k.startswith("rwkv.")}
    moba = {k[5:]: v for k, v in ckpt.items() if k.startswith("moba.")}

    n_layer = len({k.split('.')[1] for k in rwkv if k.startswith("blocks.")})
    n_embd = rwkv['emb.weight'].shape[1]
    vocab_size = rwkv['emb.weight'].shape[0]
    n_head = n_embd // 64
    n_moba_layer = len({k.split('.')[0] for k in moba if k[0].isdigit()})

    return MOBAConfig(n_moba_layer=n_moba_layer, n_head=n_head, n_embd=n_embd, moba_chunk_size=moba_chunk_size, moba_topk=moba_topk)


def load_rwkvx(rwkv_args, moba_chunk_size=2048, moba_topk=3):
    import os
    os.environ["RWKV_JIT_ON"] = "0"
    os.environ["RWKV_CUDA_ON"] = "1"
    os.environ["RWKV_V7_ON"] = "1"
    os.environ["RWKV_HEAD_SIZE_A"] = "64"

    # import RWKV and RWKVHybrid
    from src.model import RWKVHybrid, RWKV
    moba_config = load_configs_from_ckpt(rwkv_args.load_model, moba_chunk_size=moba_chunk_size, moba_topk=moba_topk)
    rwkv = RWKV(rwkv_args)
    model = RWKVHybrid(rwkv, rwkv_args, moba_config)
    # load state dict
    state_dict = torch.load(rwkv_args.load_model, map_location='cpu', weights_only=True)
    msg = model.load_state_dict(state_dict, strict=False)
    print(f'Load state dict: {msg} from {rwkv_args.load_model}')
    model = model.bfloat16()
    # load tokenizer
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer = TRIE_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")
    return model, tokenizer