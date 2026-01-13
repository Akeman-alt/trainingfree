import os
import glob
import pandas as pd
import numpy as np
import matplotlib

# 设置 matplotlib 后端，防止在没有显示器的服务器上报错
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
import sys


# ================= 配置区域 =================
# 🎯 奖励定义：必须与 flow_module.py 完全一致
TARGET_CHAR = 'A' 

def setup_logger(save_dir):
    """
    配置日志系统：同时输出到屏幕和文件
    保存路径：run_dir/analysis_result.log
    """
    log_file = os.path.join(save_dir, 'analysis_result.log')
    
    # 获取 root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清空已有的 handlers，防止重复打印
    if logger.hasHandlers():
        logger.handlers.clear()

    # 1. 文件输出 Handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)

    # 2. 屏幕输出 Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s')) # 屏幕上看着清爽点，不用时间戳
    logger.addHandler(console_handler)
    
    logger.info(f"📝 日志将自动保存至: {log_file}")
    return logger

def calculate_reward(sequence):
    """计算序列中目标氨基酸的占比"""
    if not isinstance(sequence, str) or len(sequence) == 0:
        return 0.0
    return sequence.count(TARGET_CHAR) / len(sequence)

def analyze_experiment(run_dir):
    # 初始化日志
    logger = setup_logger(run_dir)
    
    logger.info(f"🚀 正在分析目录: {run_dir}")
    logger.info(f"🎯 目标氨基酸: '{TARGET_CHAR}'")
    
    # 1. 寻找所有的 sc_results.csv
    # 你的目录结构似乎是 run_dir/length_*/sample_*/sc_results.csv
    search_pattern = os.path.join(run_dir, "length_*", "sample_*", "sc_results.csv")
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        logger.error(f"❌ 未找到结果文件，请检查路径: {search_pattern}")
        return

    data_list = []
    logger.info(f"⏳ 正在读取 {len(csv_files)} 个样本...")

    for f in csv_files:
        try:
            df = pd.read_csv(f)
            # 优先取 RMSD 最低的那条（代表该次采样的最佳结果）
            if 'bb_rmsd' in df.columns:
                best_row = df.sort_values(by='bb_rmsd', ascending=True).iloc[0]
            else:
                best_row = df.iloc[0]

            # 获取序列 (兼容不同列名)
            sequence = best_row.get('sequence', '')
            if not isinstance(sequence, str):
                # 如果 csv 里没序列，尝试去读 fasta
                sample_dir = os.path.dirname(f)
                codesign_path = os.path.join(sample_dir, "self_consistency", "codesign_seqs", "codesign.fa")
                if os.path.exists(codesign_path):
                    with open(codesign_path, 'r') as fa:
                        lines = fa.readlines()
                        if len(lines) >= 2: sequence = lines[1].strip()

            # 计算奖励
            reward = calculate_reward(sequence)
            rmsd = best_row.get('bb_rmsd', np.nan)

            data_list.append({
                'reward': reward,
                'rmsd': rmsd,
                'sequence': sequence,
                'length': len(sequence)
            })
            
        except Exception as e:
            logger.warning(f"⚠️ 读取文件出错 {f}: {e}")
            continue

    df_all = pd.DataFrame(data_list)
    
    if len(df_all) == 0:
        logger.error("⚠️ 没有有效数据。")
        return

    # 2. 统计结果
    avg_reward = df_all['reward'].mean()
    avg_rmsd = df_all['rmsd'].mean()
    max_reward = df_all['reward'].max()

    logger.info("\n" + "="*50)
    logger.info("       🧪 实验结果最终核对       ")
    logger.info("="*50)
    logger.info(f"【Reward】 (A 的占比)")
    logger.info(f"  平均值 : {avg_reward:.2%} (Baseline通常 < 10%)")
    logger.info(f"  最大值 : {max_reward:.2%}")
    logger.info("-" * 50)
    logger.info(f"【RMSD】 (结构稳定性)")
    logger.info(f"  平均值 : {avg_rmsd:.4f} Å")
    logger.info("="*50)

    # 3. 👁️ 视觉核对：打印 Top 3 序列
    logger.info("\n🏆【Top 3 高分序列展示】")
    top_seqs = df_all.sort_values(by='reward', ascending=False).head(3)
    for i, row in top_seqs.iterrows():
        seq = row['sequence']
        display_seq = seq[:50] + "..." if len(seq) > 50 else seq
        logger.info(f"Runs: {row['reward']:.2%} | Seq: {display_seq}")

    # 4. 绘图
    # 图片保存到 run_dir 下，而不是当前代码目录，防止覆盖
    plot_path = os.path.join(run_dir, "check_reward_dist.png")
    
    plt.figure(figsize=(10, 4))
    sns.histplot(df_all['reward'], bins=20, kde=True, color='green')
    plt.title(f'Distribution of Alanine (A) Content\nMean: {avg_reward:.2%}')
    plt.xlabel('Fraction of A')
    plt.axvline(0.08, color='red', linestyle='--', label='Natural Baseline')
    plt.legend()
    plt.savefig(plot_path)
    plt.close() # 关闭画布，释放内存
    
    logger.info(f"\n📊 分布图已保存: {plot_path}")

if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description="Analyze MultiFlow guidance experiment results.")
    parser.add_argument('--run_dir', type=str, required=True, help="Path to the experiment run directory (e.g., .../run_2025-...)")
    
    args = parser.parse_args()
    
    # 检查路径是否存在
    if not os.path.exists(args.run_dir):
        print(f"Error: Directory not found: {args.run_dir}")
    else:
        analyze_experiment(args.run_dir)