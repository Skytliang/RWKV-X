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
    parser.add_argument('--device', type=str, default='cuda:0')
    # add a group for moba
    group = parser.add_argument_group('moba')
    group.add_argument('--moba_chunk_size', type=int, default=2048, help='chunk size for moba')
    group.add_argument('--moba_topk', type=int, default=3, help='topk for moba')
    # add a group for plot
    group = parser.add_argument_group('plot')
    group.add_argument('--heatmap_data', type=str, default='')

    args = parser.parse_args()
    return args


def generate_prompt_landmark(n_garbage, seed, n_garbage_prefix):
    """Generates a text file and inserts an passkey at a random position."""
    rnd_state = random.get_state()
    random.seed(seed)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 10000)
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

@torch.inference_mode()
def passkey_retrieval_test(model, tokenizer, device, n_garbage_prefix, n_garbage=60000, seed=666):
    prompt, answer = generate_prompt_landmark(n_garbage, seed, n_garbage_prefix)
    input_ids = tokenizer.encode(prompt)
    len_token = len(input_ids)
    answer_ids = tokenizer.encode(answer)
    
    # generate answer
    gen_length = len(answer_ids)
    all_outputs = []
    x = torch.tensor([input_ids], device=device, dtype=torch.long)
    for i in range(gen_length):
        logits = model.forward(x)
        last_logit = logits[:, -1:, :]
        next_token = torch.argmax(last_logit, dim=-1)
        x = torch.cat([x, next_token], dim=1)
        all_outputs.append(next_token.item())

    
    model_answer = tokenizer.decode(all_outputs).strip()
    gold_answer = tokenizer.decode(answer_ids).strip()
    #print(f'model_answer: {model_answer}, gold_answer: {gold_answer}')
    is_correct = (model_answer == gold_answer)
    return is_correct, len_token


def plot_heatmap(df, args, fontsize=30):
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
    #plt.title('Needle in a Haystack Evaluation', fontsize=24)

    # 替换 X 轴标签为形如 '4k', '8k' 的格式
    xticks = ax.get_xticks()
    xtick_labels = pivot_table.columns.tolist()
    xtick_labels_formatted = [f"{int(x)//1000}K" for x in xtick_labels]
    ax.set_xticklabels(xtick_labels_formatted, fontsize=fontsize)  # Set x-axis labels with larger font

    # More aesthetics
    plt.xlabel('Context Length', fontsize=fontsize+6)  # X-axis label with larger font
    plt.ylabel('Answer Depth (%)', fontsize=fontsize+6)  # Y-axis label with larger font
    #plt.xticks(rotation=45, fontsize=16)  # Rotate and enlarge x-axis labels
    plt.yticks(rotation=0, fontsize=fontsize)  # Enlarge y-axis labels
    # 设置 colorbar 字体大小
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fontsize+6)  # 增大 colorbar 的刻度字体
    cbar.set_label('Score', fontsize=fontsize+6)  # 增大 colorbar 的标题字体
    plt.tight_layout()  # Fits everything neatly into the figure area
    # save
    log_dir = Path(args.log_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(args.base_model).stem
    output_stem = log_dir / f"{base_name}_heatmap_{args.max_tokens}"
    plt.savefig(f"{output_stem}.png", dpi=300, bbox_inches='tight')
    return output_stem


def main(args):
    from load_utils import load_rwkvx
    device = torch.device(args.device)
    torch.cuda.set_device(args.device)

    # Load model and tokenizer 
    model, tokenizer = load_rwkvx(
        args.base_model, device=device, 
        moba_chunk_size=args.moba_chunk_size, moba_topk=args.moba_topk
        )

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
                torch.cuda.empty_cache()
            avg_tokens = total_tokens//args.num_tests if avg_tokens is None else avg_tokens
            accuracy = float(passed_tests)/args.num_tests
            depth = n_garbage_prefix/n_garbage
            #print("accuracy on the token length %d, depth %f, is %f"%(avg_tokens,depth, accuracy))
            result = {"Context Length": context_length, "Document Depth": round(depth*100, -1),"Score": accuracy * 100}
            all_accuries.append(result)
    df = pd.DataFrame(all_accuries)
    # plot heatmap
    output_stem = plot_heatmap(df, args)
    df.to_csv(f"{output_stem}.csv", index=False)
    
    
if __name__ == "__main__":
    args = parse_config()
    if args.heatmap_data:
        df = pd.read_csv(args.heatmap_data)
        plot_heatmap(df, args)
    else:
        main(args)