import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import seaborn as sns

# ================= 配置区域 =================
# 🔴 请替换为你最新的 run 目录路径
#RUN_DIR = "/data2/zq/multiflow/inference_outputs/weights/last/unconditional/run_2025-12-23_16-57-36"
RUN_DIR = "/data2/zq/multiflow/inference_outputs/weights/last/unconditional/run_2025-12-23_23-39-18"
# 🎯 奖励定义：必须与 flow_module.py 完全一致
# 你的训练代码：target_aa_id = 0 (即 'A')
TARGET_CHAR = 'A' 

def calculate_reward(sequence):
    """计算序列中目标氨基酸的占比"""
    if not isinstance(sequence, str) or len(sequence) == 0:
        return 0.0
    # 统计 'A' 的数量 / 总长度
    return sequence.count(TARGET_CHAR) / len(sequence)
# ===========================================

def analyze_experiment(run_dir):
    print(f"🚀 正在分析目录: {run_dir}")
    print(f"🎯 目标氨基酸: '{TARGET_CHAR}' (对应 ID=0)")
    
    # 1. 寻找所有的 sc_results.csv
    search_pattern = os.path.join(run_dir, "length_*", "sample_*", "sc_results.csv")
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print("❌ 未找到结果文件，请检查路径。")
        return

    data_list = []
    
    print(f"⏳ 正在读取 {len(csv_files)} 个样本...")

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
            
        except Exception:
            continue

    df_all = pd.DataFrame(data_list)
    
    if len(df_all) == 0:
        print("⚠️ 没有有效数据。")
        return

    # 2. 统计结果
    avg_reward = df_all['reward'].mean()
    avg_rmsd = df_all['rmsd'].mean()

    print("\n" + "="*50)
    print("       🧪 实验结果最终核对       ")
    print("="*50)
    print(f"【Reward】 (A 的占比)")
    print(f"  平均值 : {avg_reward:.2%} (Baseline通常 < 10%)")
    print(f"  最大值 : {df_all['reward'].max():.2%}")
    print("-" * 50)
    print(f"【RMSD】 (结构稳定性)")
    print(f"  平均值 : {avg_rmsd:.4f} Å")
    print("="*50)

    # 3. 👁️ 视觉核对：打印 Top 3 序列
    print("\n🏆【Top 3 高分序列展示】(请人眼检查是否有很多 A)")
    top_seqs = df_all.sort_values(by='reward', ascending=False).head(3)
    for i, row in top_seqs.iterrows():
        seq = row['sequence']
        # 为了显示方便，截取前 50 个字符
        display_seq = seq[:50] + "..." if len(seq) > 50 else seq
        print(f"Runs: {row['reward']:.2%} | Seq: {display_seq}")

    # 4. 绘图
    plt.figure(figsize=(10, 4))
    sns.histplot(df_all['reward'], bins=20, kde=True, color='green')
    plt.title(f'Distribution of Alanine (A) Content\nMean: {avg_reward:.2%}')
    plt.xlabel('Fraction of A')
    plt.axvline(0.08, color='red', linestyle='--', label='Natural Baseline')
    plt.legend()
    plt.savefig("check_reward_base.png")
    print(f"\n📊 分布图已保存: check_reward.png")

if __name__ == "__main__":
    analyze_experiment(RUN_DIR)