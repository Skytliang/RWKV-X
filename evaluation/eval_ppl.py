import argparse
import datasets
import gc
import sys
import torch
import warnings
from transformers import AutoTokenizer
from tqdm import tqdm
import pdb

def compute_perplexity(
    encodings, model, tokenizer, add_start_token: bool = True, device=None, max_length=None, sliding_window=256, truncate=False, aggressive_memory=False, hide_progress=False, delta_ratio=None
):
    r"""Compute "sliding window" perplexity on a dataset. Validated against the calculations reported in arXiv 2306.15595"""
    if device is not None:
        assert device in ["gpu", "cpu",
                          "cuda"], "device should be either gpu or cpu."
        if device == "gpu":
            device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if add_start_token:
        # leave room for <BOS> token to be added:
        # assert (
        #     tokenizer.bos_token is not None
        # ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]
    

    if max_length and truncate:
        encoded_texts = [x[0:max_tokenized_len] for x in encoded_texts]
        attn_masks = [x[0:max_tokenized_len] for x in attn_masks]
        sliding_window = max_tokenized_len

    pbar = tqdm(total=len(encoded_texts), disable=hide_progress)
    nlls = []
    for encoding_index in range(0, len(encoded_texts)):

        labels = torch.tensor(encoded_texts[encoding_index:encoding_index+1])
        seq_len = labels.size(1)

        prev_end_loc = 0
        for begin_loc in range(0, seq_len, sliding_window):

            end_loc = min(begin_loc + max_tokenized_len, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = labels[:, begin_loc:end_loc].to(device)
            if add_start_token:
                bos_tokens_tensor = torch.tensor(
                    [[0]] * input_ids.size(dim=0)).to(device)
                input_ids = torch.cat(
                    [bos_tokens_tensor, input_ids], dim=1)

            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            with torch.no_grad():
                
                input_ids = input_ids[0].tolist()
                
                # chunkwise prefill
                CHUNK_SIZE = 4096
                state = None
                all_logits = None
                for i in range(0, len(input_ids), CHUNK_SIZE):
                    prefill_token = input_ids[i: i+CHUNK_SIZE]
                    logits, state = model(prefill_token, state, full_output=True)
                    if all_logits is None:
                        all_logits = logits
                    else:
                        all_logits = torch.cat((all_logits, logits), dim=0)
                
                # non-chunkwise prefill
                # logits, _ = model(input_ids, None, full_output=True)
                
                logits = all_logits[..., :-1, :]
                # pdb.set_trace()
                target_ids = target_ids[..., 1:].contiguous()
                neg_log_likelihood = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), target_ids.view(-1), reduction='mean')
            
            if aggressive_memory:
                outputs = None
                input_ids = None
                target_ids = None
                gc.collect()
                torch.cuda.empty_cache()

            nlls.append(neg_log_likelihood)

            ppl = float(torch.exp(torch.stack(nlls).mean()).float().cpu())
            pbar.set_postfix(ppl=ppl)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        pbar.update(1)

    ppl = float(torch.exp(torch.stack(nlls).mean()).float().cpu())
    return {"mean_perplexity": ppl}


def main(args):
    models = [x[0] for x in args.model]
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    tokenizer.pad_token = tokenizer.eos_token

    if args.tokenized:
        try:
            input_texts = datasets.load_from_disk(args.tokenized)
        except:
            input_texts = datasets.load_dataset(
                args.tokenized, name=args.subset, split=args.split)
    else:
        input_texts = datasets.load_dataset(
            args.dataset, name=args.subset, split=args.split)

        def tokenize(example):
            tokenized = tokenizer(
                example[args.feature],
                add_special_tokens=False,
                padding=True,
                truncation=False,
                max_length=sys.maxsize,
                return_attention_mask=True,
            )
            example["input_ids"] = tokenized["input_ids"]
            example["attention_mask"] = tokenized["attention_mask"]
            example["tokenized_len"] = len(tokenized["input_ids"])
            return example

        input_texts = input_texts.map(tokenize, num_proc=64)
        if args.save_tokenized:
            from datasets import DatasetDict
            dataset = DatasetDict({"test": input_texts})
            dataset.push_to_hub(args.save_tokenized)
            print(f"Saved tokenized dataset to {args.save_tokenized}")
            return

    if args.dataset_min_tokens:
        input_texts = input_texts.filter(
            lambda x: x["tokenized_len"] >= args.dataset_min_tokens, num_proc=64)
    if args.samples:
        input_texts = input_texts[:args.samples]

    if args.tokens_step:
        tokens = [x for x in range(
            args.min_tokens, args.max_tokens + 1, args.tokens_step)]
    else:
        tokens = [args.min_tokens]
        while args.min_tokens < args.max_tokens:
            point = tokens[-1] * 2
            if point <= args.max_tokens:
                tokens.append(point)
            else:
                break

    results = []
    for model in tqdm(models, desc="Model", leave=False, disable=args.hide_progress):
        torch.cuda.empty_cache()
        
        # load RWKV model
        import os
        os.environ["RWKV_JIT_ON"] = "1"
        os.environ["RWKV_CUDA_ON"] = "1"
        os.environ["RWKV_V7_ON"] = '1'
        
        from rwkv.model import RWKV
        # from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
        loaded = RWKV(model=model, strategy="cuda fp16")
        # tokenizer = TRIE_TOKENIZER("rwkv_vocab_v20230424")
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file("20B_tokenizer.json")
        
        result = []
        for max_length in tokens:
            ppl = compute_perplexity(model=loaded, tokenizer=tokenizer, encodings=input_texts,
                                     add_start_token=True, max_length=max_length,
                                     sliding_window=args.sliding_window, truncate=args.truncate,
                                     aggressive_memory=args.aggressive_memory, hide_progress=args.hide_progress, delta_ratio=args.delta_ratio)['mean_perplexity']
            print(f"{model}: {max_length}={ppl}")
            result.append(ppl)

        result.insert(0, model)
        results.append(result)

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(f",{','.join([str(x) for x in tokens])}\n")
            for result in results:
                f.write(f"{','.join([str(x) for x in result])}\n")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", action="append", nargs="+")
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-s", "--subset", type=str)
    parser.add_argument("-f", "--feature", type=str)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--min-tokens", type=int, default=256)
    parser.add_argument("--dataset-min-tokens", type=int)
    parser.add_argument("--tokens-step", type=int)
    parser.add_argument("--sliding-window", type=int, default=256)
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--save-tokenized", type=str)
    parser.add_argument("--tokenized", type=str)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--aggressive-memory", action="store_true")
    parser.add_argument("--hide-progress", action="store_true")
    parser.add_argument("--delta_ratio", type=float)
    main(parser.parse_args())
