import pandas as pd
import argparse
import os

# 任务分类定义
TASK_CATEGORIES = {
    "Single-Document QA": {
        "en": ["longbench_narrativeqa", "longbench_qasper", "longbench_multifieldqa_en"],
        "zh": ["longbench_multifieldqa_zh"]
    },
    "Multi-Document QA": {
        "en": ["longbench_hotpotqa", "longbench_2wikimqa", "longbench_musique"],
        "zh": ["longbench_dureader"]
    },
    "Summarization": {
        "en": ["longbench_gov_report", "longbench_qmsum", "longbench_multi_news"],
        "zh": ["longbench_vcsum"]
    },
    "Few-shot Learning": {
        "en": ["longbench_trec", "longbench_triviaqa", "longbench_samsum"],
        "zh": ["longbench_lsht"]
    },
    "Synthetic Tasks": {
        "en": ["longbench_passage_count", "longbench_passage_retrieval_en"],
        "zh": ["longbench_passage_retrieval_zh"]
    },
    "Code Completion": {
        "en": ["longbench_lcc", "longbench_repobench-p"],
        "zh": []
    }
}

def analyze(file_path, output_dir):
    df = pd.read_csv(file_path)

    # 1. 提取每个任务的得分
    task_scores = {}
    for col in df.columns:
        values = pd.to_numeric(df[col], errors='coerce').dropna()
        nonzero_values = values[values != 0]
        if not nonzero_values.empty:
            task_scores[col] = nonzero_values.iloc[0] * 100

    # 2. 构建详细得分数据
    detailed_scores = []
    for category, langs in TASK_CATEGORIES.items():
        for lang, tasks in langs.items():
            for task in tasks:
                if task in task_scores:
                    detailed_scores.append((task, category, lang, round(task_scores[task], 1)))

    detailed_df = pd.DataFrame(detailed_scores, columns=["任务", "任务类别", "语言", "得分"])

    # 3. 汇总平均值
    summary_results = []

    for category, langs in TASK_CATEGORIES.items():
        for lang, tasks in langs.items():
            scores = [task_scores[task] for task in tasks if task in task_scores]
            if scores:
                avg = sum(scores) / len(scores)
                summary_results.append((category, lang, "平均值", round(avg, 1)))

    # 总体统计
    en_scores = [score for _, _, lang, score in detailed_scores if lang == "en"]
    zh_scores = [score for _, _, lang, score in detailed_scores if lang == "zh"]
    all_scores = en_scores + zh_scores

    if en_scores:
        summary_results.append(("全部任务", "en", "平均值", round(sum(en_scores) / len(en_scores), 1)))
    if zh_scores:
        summary_results.append(("全部任务", "zh", "平均值", round(sum(zh_scores) / len(zh_scores), 1)))
    if all_scores:
        summary_results.append(("全部任务", "all", "平均值", round(sum(all_scores) / len(all_scores), 1)))


    summary_df = pd.DataFrame(summary_results, columns=["任务类别", "语言", "说明", "平均得分"])

    # 4. 自动命名并保存输出
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    summary_path = os.path.join(output_dir, f"{base_name}_summary.csv")
    detail_path = os.path.join(output_dir, f"{base_name}_detail.csv")

    summary_df.to_csv(summary_path, index=False)
    detailed_df.to_csv(detail_path, index=False)

    print("✅ 分析完成！输出文件：")
    print(f" - 任务汇总得分：{summary_path}")
    print(f" - 任务详细得分：{detail_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="📊 LongBench 任务分析工具")
    parser.add_argument("file", help="输入 CSV 文件路径")
    parser.add_argument("-o", "--output", default=".", help="输出目录（默认当前目录）")
    args = parser.parse_args()

    analyze(args.file, args.output)
