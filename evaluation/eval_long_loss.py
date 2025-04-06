import torch
from tqdm import tqdm
import json
import importlib
import sys
import os
import numpy as np
import pandas as pd
import argparse


def load_jsonl(filename):
   with open(filename, 'r', encoding='utf-8') as f:
       return [json.loads(line) for line in f]


def load_rwkv(config, strategy="cuda fp16"):
    path = config['model']
    vocab = config['tokenizer']
    import os

    os.environ["RWKV_JIT_ON"] = "1"
    os.environ["RWKV_CUDA_ON"] = "1"
    os.environ["RWKV_V7_ON"] = "1"
    path = path.replace('.pth', '')
    
    if "rwkv.model" in sys.modules:
        importlib.reload(sys.modules["rwkv.model"])
    from rwkv.model import RWKV
    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
    model = RWKV(model=path, strategy=strategy)
    
    if vocab == "rwkv_vocab_v20230424.txt":
        tokenizer = TRIE_TOKENIZER("rwkv_vocab_v20230424.txt")
    elif vocab == '20B_tokenizer.json':
        # 20B tokenizer
        from rwkv.utils import PIPELINE
        pipeline = PIPELINE(model, "20B_tokenizer.json")
        tokenizer = pipeline.tokenizer
        
    return model, tokenizer


def load_hf(config):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(config['model'], 
                                                device_map="cuda:0", 
                                                trust_remote_code=True, 
                                                cache_dir='rwkv_model').eval()
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'], trust_remote_code=True, cache_dir='rwkv_model')
    
    return model, tokenizer


def load_mamba(config):
    # pip install mamba-ssm
    # pip install causal-conv1d>=1.2.0
    from transformers import AutoTokenizer
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
    model = MambaLMHeadModel.from_pretrained(config['model'], device="cuda", dtype=torch.float16)
    model.device = torch.device("cuda")

    return model, tokenizer

def load_rwkvx(config):
    import os
    os.environ["RWKV_JIT_ON"] = "0"
    os.environ["RWKV_CUDA_ON"] = "1"
    os.environ["RWKV_V7_ON"] = "1"
    os.environ["RWKV_HEAD_SIZE_A"] = "64"

    # import RWKV and RWKVHybrid
    from src.model import RWKVHybrid, RWKV
    from utils import load_configs_from_ckpt
    rwkv_config, moba_config = load_configs_from_ckpt(config['model'])
    rwkv = RWKV(rwkv_config)
    model = RWKVHybrid(rwkv, rwkv_config, moba_config)
    # load state dict
    state_dict = torch.load(config['model'], map_location='cpu', weights_only=True)
    msg = model.load_state_dict(state_dict, strict=False)
    print(f'Load state dict: {msg} from {config["model"]}')
    model = model.bfloat16().cuda()
    # load tokenizer
    from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
    tokenizer = TRIE_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")
    return model, tokenizer

def save_mean_loss(loss_vectors, save_path):
    import numpy as np
    loss_array = np.array(loss_vectors)
    mean_loss = np.mean(loss_array, axis=0)
    np.save(save_path, mean_loss)
    return mean_loss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--model_type', type=str, default='rwkv')
    parser.add_argument('--tokenizer', type=str, default='rwkv_vocab_v20230424.txt')
    parser.add_argument('--dataset', type=str, default='pg19_test_original.jsonl')
    parser.add_argument('--log_name', type=str, default='logs/pile_pg19/')
    parser.add_argument('--max_samples', type=int, default=100)
    parser.add_argument('--begin_token', type=int, default=2048)
    parser.add_argument('--seq_length', type=int, default=32768)
    return parser.parse_args()

args = parse_args()
dataset = load_jsonl(args.dataset)
model_list = [{'type': args.model_type, 'model': args.model, 'tokenizer': args.tokenizer}]
# model_list = [
#     # {'type': 'rwkv', 'model': 'rwkv_model/RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth', 'tokenizer': 'rwkv_vocab_v20230424.txt'},
#     # {'type': 'rwkv', 'model': "rwkv_model/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth", 'tokenizer': 'rwkv_vocab_v20230424.txt'},
#     # {'type': 'rwkv', 'model': "rwkv_model/RWKV-5-World-3B-v2-20231113-ctx4096.pth", 'tokenizer': 'rwkv_vocab_v20230424.txt'},
#     # {'type': 'rwkv', 'model': "rwkv_model/RWKV-4-World-3B-v1-20230619-ctx4096.pth", 'tokenizer': 'rwkv_vocab_v20230424.txt'},
#     # {'type': 'rwkv', 'model': "rwkv_model/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth", 'tokenizer': 'rwkv_vocab_v20230424.txt'},
#     # {'type': 'hf', 'model': 'SmerkyG/RWKV7-2.9B-World3-128k-250225', 'tokenizer': 'SmerkyG/RWKV7-2.9B-World3-128k-250225'},
    
#     # {'type': 'rwkv', 'model': "rwkv_model/RWKV-x070-Pile-168M-20241120-ctx4096.pth", 'tokenizer': '20B_tokenizer.json'},
#     # {'type': 'rwkv', 'model': "rwkv_model/rwkv-x060-173m-pile-20240515-ctx4k.pth", 'tokenizer': '20B_tokenizer.json'},
#     # {'type': 'rwkv', 'model': "rwkv_model/RWKV-4-Pile-169M-20220807-8023.pth", 'tokenizer': '20B_tokenizer.json'},
    
#     # {'type': 'rwkv', 'model': "rwkv_model/RWKV-x070-Pile-1.47B-20241210-ctx4096.pth", 'tokenizer': '20B_tokenizer.json'},
    
#     # {'type': 'hf_mamba', 'model': 'state-spaces/mamba-2.8b-hf', 'tokenizer': 'state-spaces/mamba-2.8b-hf'},
#     # {'type': 'hf_mamba', 'model': 'state-spaces/mamba-130m-hf', 'tokenizer': 'state-spaces/mamba-130m-hf'},
# ]
log_name_prefix = args.log_name
if not os.path.exists(log_name_prefix):
    os.makedirs(log_name_prefix)

seq_length_list = [i for i in range(2048, args.seq_length, 2048)] + [args.seq_length]
max_samples = args.max_samples
begin_token = args.begin_token


for config in model_list:
    # load model
    model_path = config['model']
    print(f'Loading model: {model_path}')
    if config['type'] == 'rwkv':
        model, tokenizer = load_rwkv(config)
    elif config['type'] == 'rwkvx':
        model, tokenizer = load_rwkvx(config)
    elif config['type'] in ['hf', 'hf_mamba']:
        model, tokenizer = load_hf(model_path)
    elif config['type'] == 'mamba':
        model, tokenizer = load_mamba(config)
    else:
        raise ValueError(f'Unknown model type: {config["type"]}')

    seq_length2loss = {}
    for seq_length in tqdm(seq_length_list, desc='Seq length'):
        tested_samples = 0
        all_losses = []
        for sample in dataset:
            # tokenize
            if config['type'] == 'rwkv' or config['type'] == 'rwkvx':
                if config['tokenizer'] == "rwkv_vocab_v20230424.txt":
                    input_ids = tokenizer.encode(sample['text'])
                elif config['tokenizer'] == '20B_tokenizer.json':
                    input_ids = tokenizer.encode(sample['text']).ids  # 20B tokenizer
            elif config['type'] in ['hf', 'hf_mamba', 'mamba']:
                input_ids = tokenizer.encode(sample['text'])
            else:
                raise ValueError(f'Unknown model type: {config["type"]}')
            
            if len(input_ids) - begin_token < seq_length:
                continue
            
            # input_ids = input_ids[:seq_length]
            input_ids = input_ids[begin_token: seq_length+begin_token+1]  # ignore the first 2048 tokens, for PG19

            # chunk-wise prefill and compute loss
            CHUNK_SIZE = 4096 + 1
            state = None
            chunk_losses = []
            last_token = None 
            
            with torch.inference_mode():
                if config['type'] in ['hf_mamba', 'mamba', 'rwkvx']:
                    result = model.forward(torch.tensor([input_ids]).cuda())
                    logits = result.logits.squeeze(0) if config['type'] != 'rwkvx' else result.squeeze(0)
                    labels = torch.tensor(input_ids[1:], device=logits.device)
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    loss = loss_fct(logits[:-1].view(-1, logits.size(-1)), labels.view(-1))
                    chunk_losses.append(loss.cpu())
                else:
                    for i in range(0, len(input_ids), CHUNK_SIZE):
                        chunk_tokens = input_ids[i: i+CHUNK_SIZE]
                        
                        if config['type'] == 'rwkv':
                            logits, state = model(chunk_tokens, state, full_output=True)
                        elif config['type'] == 'hf':
                            result = model.forward(torch.tensor([chunk_tokens]).cuda(), use_cache=True, past_key_values=state)
                            logits = result.logits.squeeze(0)
                            state = result.past_key_values
                        else:
                            raise ValueError(f'Unknown model type: {config["type"]}')
                        
                        if len(chunk_tokens) > 1:
                            if last_token is not None:
                                labels = torch.tensor([chunk_tokens[0]], device=logits.device)
                                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                                loss = loss_fct(last_token.view(-1, last_token.size(-1)), labels.view(-1))
                                chunk_losses.append(loss.cpu())
                                
                            labels = torch.tensor(chunk_tokens[1:], device=logits.device)
                            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                            loss = loss_fct(logits[:-1].view(-1, logits.size(-1)), labels.view(-1))
                            chunk_losses.append(loss.cpu())
                            
                            last_token = logits[-1:]
                
                sample_losses = torch.cat(chunk_losses)
                all_losses.append(sample_losses)
                    
            tested_samples += 1
            
            if tested_samples >= max_samples:
                break

        all_losses = np.array([loss.float() for loss in all_losses])
        mean_loss = np.mean(all_losses, axis=1) # (num_samples,)
        seq_length2loss[seq_length] = mean_loss.mean()
    file_name = f'{log_name_prefix}{model_path.split("/")[-1].replace(".pth", "")}.csv'
    df = pd.DataFrame(seq_length2loss.items(), columns=['seq_length', 'mean_loss'])
    df.to_csv(file_name, index=False)
    print('-------------------------------------')
    print(f'model name: {config["model"]}')
    print(f'Saved to {file_name}')
    print('-------------------------------------')
