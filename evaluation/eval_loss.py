import torch
from tqdm import tqdm
import json
import importlib
import sys
import os


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


def save_mean_loss(loss_vectors, save_path):
    import numpy as np
    loss_array = np.array(loss_vectors)
    mean_loss = np.mean(loss_array, axis=0)
    np.save(save_path, mean_loss)
    return mean_loss


# dataset = load_jsonl('datasets/proof_pile_test_original.jsonl')
dataset = load_jsonl('pg19_test_original.jsonl')
model_list = [
    # {'type': 'rwkv', 'model': 'rwkv_model/RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth', 'tokenizer': 'rwkv_vocab_v20230424.txt'},
    # {'type': 'rwkv', 'model': "rwkv_model/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth", 'tokenizer': 'rwkv_vocab_v20230424.txt'},
    # {'type': 'rwkv', 'model': "rwkv_model/RWKV-5-World-3B-v2-20231113-ctx4096.pth", 'tokenizer': 'rwkv_vocab_v20230424.txt'},
    # {'type': 'rwkv', 'model': "rwkv_model/RWKV-4-World-3B-v1-20230619-ctx4096.pth", 'tokenizer': 'rwkv_vocab_v20230424.txt'},
    {'type': 'rwkv', 'model': "rwkv_model/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth", 'tokenizer': 'rwkv_vocab_v20230424.txt'},
    # {'type': 'hf', 'model': 'SmerkyG/RWKV7-2.9B-World3-128k-250225', 'tokenizer': 'SmerkyG/RWKV7-2.9B-World3-128k-250225'},
    
    # {'type': 'rwkv', 'model': "rwkv_model/RWKV-x070-Pile-168M-20241120-ctx4096.pth", 'tokenizer': '20B_tokenizer.json'},
    # {'type': 'rwkv', 'model': "rwkv_model/rwkv-x060-173m-pile-20240515-ctx4k.pth", 'tokenizer': '20B_tokenizer.json'},
    # {'type': 'rwkv', 'model': "rwkv_model/RWKV-4-Pile-169M-20220807-8023.pth", 'tokenizer': '20B_tokenizer.json'},
    
    # {'type': 'rwkv', 'model': "rwkv_model/RWKV-x070-Pile-1.47B-20241210-ctx4096.pth", 'tokenizer': '20B_tokenizer.json'},
    
    # {'type': 'hf_mamba', 'model': 'state-spaces/mamba-2.8b-hf', 'tokenizer': 'state-spaces/mamba-2.8b-hf'},
    # {'type': 'hf_mamba', 'model': 'state-spaces/mamba-130m-hf', 'tokenizer': 'state-spaces/mamba-130m-hf'},
]
log_name_prefix = 'logs/pile_pg19_20250304/'
if not os.path.exists(log_name_prefix):
    os.makedirs(log_name_prefix)
seq_length = 32768 + 1
max_samples = 100
begin_token = 2048


for config in model_list:
    # load model
    model_path = config['model']
    print(f'Loading model: {model_path}')
    if config['type'] == 'rwkv':
        model, tokenizer = load_rwkv(config)
    elif config['type'] in ['hf', 'hf_mamba']:
        model, tokenizer = load_hf(model_path)
    elif config['type'] == 'mamba':
        model, tokenizer = load_mamba(config)
    else:
        raise ValueError(f'Unknown model type: {config["type"]}')
        
    all_losses = []
    tested_samples = 0
    pbar = tqdm(dataset, total=min(len(dataset), max_samples))
    with torch.no_grad():
        for sample in dataset:
            
            # tokenize
            if config['type'] == 'rwkv':
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
            input_ids = input_ids[begin_token: seq_length+begin_token]  # ignore the first 2048 tokens, for PG19

            # chunk-wise prefill and compute loss
            CHUNK_SIZE = 4096 + 1
            state = None
            chunk_losses = []
            last_token = None 
            
            if config['type'] in ['hf_mamba', 'mamba']:
                result = model.forward(torch.tensor([input_ids]).cuda())
                logits = result.logits.squeeze(0)
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
            pbar.update(1)
            pbar.set_description(f'Tested samples: {tested_samples}')
            
            if tested_samples >= max_samples:
                break
                
        pbar.close()
        print(f'Tested samples: {tested_samples}')

    file_name = f'{log_name_prefix}{model_path.split("/")[-1].replace(".pth", "")}.npy'
    mean_loss = save_mean_loss(all_losses, file_name)
    print('-------------------------------------')
    print(f'model name: {config["model"]}')
    print(f'context length: {seq_length}, mean loss: {mean_loss.mean()}, ppl: {2**mean_loss.mean()}')
    print(f'Saved to {file_name}')
    print('-------------------------------------')
