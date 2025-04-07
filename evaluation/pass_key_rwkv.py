import os
import math
import torch
import argparse
import random
import numpy as np
from numpy import random
from tqdm import tqdm
import transformers
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import seaborn as sns
import pdb



def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('base_model', type=str)
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--max_tokens', type=int, default=32000, help='maximum token length for evaluation')
    parser.add_argument('--interval', type=int, default=2000, help='interval for evaluation')
    parser.add_argument('--num_tests', type=int, default=5, help='number of repeat testing for each length')
    parser.add_argument('--log_name', type=str, default='logs/passkey/')

    args = parser.parse_args()
    return args


def generate_prompt_landmark(n_garbage, seed, n_garbage_prefix):
    """Generates a text file and inserts an passkey at a random position."""
    rnd_state = random.get_state()
    random.seed(seed)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 5000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question,
    ]
    random.set_state(rnd_state)
    return "\n".join(lines), str(pass_key)


def passkey_retrieval_test(model, tokenizer, device, n_garbage_prefix, n_garbage=60000, seed=666):
    prompt, answer = generate_prompt_landmark(n_garbage, seed, n_garbage_prefix)
    input_ids = tokenizer.encode(prompt)
    len_token = len(input_ids)
    answer_ids = tokenizer.encode(answer)
    
    # chunkwise prefill
    CHUNK_SIZE = 4000
    prefill_ids, next_token = input_ids[:-1], input_ids[-1]
    state = None
    for i in range(0, len(prefill_ids), CHUNK_SIZE):
        prefill_token = prefill_ids[i: i+CHUNK_SIZE]
        _, state = model(prefill_token, state)
    
    # generate answer
    gen_length = len(answer_ids)
    all_outputs = []
    for i in range(gen_length):
        logits, state = model([next_token], state)
        next_token = torch.argmax(logits, dim=-1).item()
        all_outputs.append(next_token)
    
    model_answer = tokenizer.decode(all_outputs).strip()
    gold_answer = tokenizer.decode(answer_ids).strip()
    #print(f'model_answer: {model_answer}, gold_answer: {gold_answer}')
    is_correct = (model_answer == gold_answer)
    return is_correct, len_token


def main(args):
    device = "cuda:0"
    torch.cuda.set_device(device)

    print("base model", args.base_model)
    # Load model and tokenizer
    import os
    os.environ["RWKV_JIT_ON"] = "0"
    os.environ["RWKV_CUDA_ON"] = "1"
    os.environ["RWKV_V7_ON"] = '1'
    
    from rwkv.model import RWKV
    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
    base_model = args.base_model.replace(".pth", "")
    model = RWKV(model=base_model, strategy="cuda fp16")
    tokenizer = TRIE_TOKENIZER("tokenizer/rwkv_vocab_v20230424.txt")

    total_test_points = args.max_tokens // args.interval
    all_accuries = []
    for i in tqdm(range(total_test_points)):
        context_length = (i + 1) * args.interval
        # This is a rough ratio to control the number of texts and tokens
        n_garbage = int(3.75 * (i + 1) * args.interval // 1000 * 1000)
        # 10 diffierent n_garbage_prefix for each n_garbage that uniformly distributed
        avg_tokens = None
        for n_garbage_prefix in range(0, n_garbage, n_garbage // 10):
            passed_tests = 0
            total_tokens = 0
            for k in range(args.num_tests):
                is_correct, len_tokens = passkey_retrieval_test(model, tokenizer, device, n_garbage_prefix, n_garbage=n_garbage, seed=k)
                passed_tests += is_correct
                total_tokens += len_tokens
            avg_tokens = total_tokens//args.num_tests if avg_tokens is None else avg_tokens
            accuracy = float(passed_tests)/args.num_tests
            depth = n_garbage_prefix/n_garbage
            #print("accuracy on the token length %d, depth %f, is %f"%(avg_tokens,depth, accuracy))
            result = {"Context Length": context_length, "Document Depth": round(depth*100, -1),"Score": accuracy * 100}
            all_accuries.append(result)
    df = pd.DataFrame(all_accuries)
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])
    
    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score")
    # Create the heatmap with better aesthetics
    plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    ax = sns.heatmap(
        pivot_table,
        vmin=0,
        vmax=100,
        fmt="g",
        cmap=cmap,
        cbar_kws={'label': 'Score'},
        linewidths=1.5,            # 设置线条宽度
        linecolor='white'          # 设置线条颜色为白色
    )

    # Title
    plt.title('Needle in a Haystack Evaluation', fontsize=24)

    # More aesthetics
    plt.xlabel('Context Length', fontsize=22)  # X-axis label with larger font
    plt.ylabel('Answer Depth (%)', fontsize=22)  # Y-axis label with larger font
    plt.xticks(rotation=45, fontsize=16)  # Rotate and enlarge x-axis labels
    plt.yticks(rotation=0, fontsize=16)  # Enlarge y-axis labels
    # 设置 colorbar 字体大小
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)  # 增大 colorbar 的刻度字体
    cbar.set_label('Score', fontsize=22)  # 增大 colorbar 的标题字体
    plt.tight_layout()  # Fits everything neatly into the figure area
    # save
    log_dir = Path(args.log_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(args.base_model).stem
    output_stem = log_dir / f"{base_name}_heatmap_{args.max_tokens}"
    plt.savefig(f"{output_stem}.png", dpi=300, bbox_inches='tight')
    df.to_csv(f"{output_stem}.csv", index=False)
    
    
if __name__ == "__main__":
    args = parse_config()
    main(args)