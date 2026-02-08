#!/bin/bash

# ================= 配置区域 =================
# 设置显卡 (根据你的情况修改，比如 "0" 或 "1")

SAMPLES=25  # 生成样本数 (建议 25 或 50，太少误差大，太多跑得慢)
GAMMA=0.1   # 序列学习率

echo "🚀 开始批量消融实验 (Ablation Study)..."
echo "----------------------------------------"

# ================= 第一组：探索 Struct Scale (力度) =================
# 已有数据: Scale=1.0 (rmsd 1.45), Scale=3.0 (rmsd 1.77, steps=5)
# 目标: 找到最佳 Sweet Spotdd

# 1. 实验 B1: 极轻微引导 (Scale=0.5) -> 验证是不是越小越好

# 2. 实验 B2: 中等偏强引导 (Scale=1.5) -> 验证 1.0 是否还能提升
echo "Running Experiment B2: Scale=1.5 (Steps=1)..."
python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_unconditional \
    inference.samples.samples_per_length=$SAMPLES \
    inference.inference_subdir="run_MPNNsoft_Struct_Scale1.5_Steps1" \
    inference.interpolant.sampling.use_ttt_guidance=True \
    inference.interpolant.guidance.task="struct_stability" \
    inference.interpolant.guidance.struct_scale=1.5 \
    inference.interpolant.guidance.gamma=$GAMMA \
    inference.interpolant.guidance.steps=1 \
    inference.interpolant.guidance.mpnn_ca_only=False

python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_unconditional \
    inference.samples.samples_per_length=$SAMPLES \
    inference.inference_subdir="run_MPNNsoft_Struct_Scale1.5_Steps1_N32" \
    inference.interpolant.sampling.use_ttt_guidance=True \
    inference.interpolant.guidance.task="struct_stability" \
    inference.interpolant.guidance.struct_scale=1.5 \
    inference.interpolant.guidance.N=32 \
    inference.interpolant.guidance.gamma=$GAMMA \
    inference.interpolant.guidance.steps=1 \
    inference.interpolant.guidance.mpnn_ca_only=False

# ================= 第二组：探索 Optimization Steps (步数) =================
# 已有数据: Steps=0 (rmsd 1.87), Steps=1 (rmsd 1.45)
# 目标: 看看多优化一步会不

echo "✅ 所有实验已完成！请使用 analyze_guidance.py 批量查看结果。"