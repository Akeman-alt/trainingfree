#!/bin/bash

# ==========================================
# 批量实验脚本：寻找最佳的 Guidance 参数组合
# 目标：Reward > 40% 且 RMSD < 2.0
# ==========================================

# 实验 A：稳健导航组 (High Steps, Low Gamma) - 我最看好这组
# 逻辑：走得慢(0.1)但走得久(20步)，方向准，不崩结构
echo "🚀 [1/3] Starting Experiment A: Gamma=0.1, Steps=20, Scale=5.0"
python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_unconditional \
    inference.num_gpus=4 \
    inference.samples.samples_per_length=25 \
    inference.inference_subdir="run_expA_scale5_gamma0.1_steps20" \
    inference.interpolant.guidance.struct_scale=5.0 \
    inference.interpolant.guidance.gamma=0.1 \
    inference.interpolant.guidance.steps=20 \
    inference.interpolant.guidance.lambda_kl=0.0

# 实验 B：折中减力组 (Medium Gamma, Lower Scale)
# 逻辑：步长适中(0.25)，但推力减半(2.5)，防止拉断
echo "🚀 [2/3] Starting Experiment B: Gamma=0.25, Steps=20, Scale=2.5"
python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_unconditional \
    inference.num_gpus=4 \
    inference.samples.samples_per_length=25 \
    inference.inference_subdir="run_expB_scale2.5_gamma0.25_steps20" \
    inference.interpolant.guidance.struct_scale=2.5 \
    inference.interpolant.guidance.gamma=0.25 \
    inference.interpolant.guidance.steps=20 \
    inference.interpolant.guidance.lambda_kl=0.0

# 实验 C：轻推高分组 (High Gamma, Low Scale)
# 逻辑：目标很远(0.5)，但推力极小(1.5)，以柔克刚
echo "🚀 [3/3] Starting Experiment C: Gamma=0.5, Steps=20, Scale=1.5"
python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_unconditional \
    inference.num_gpus=4 \
    inference.samples.samples_per_length=25 \
    inference.inference_subdir="run_expC_scale1.5_gamma0.5_steps20" \
    inference.interpolant.guidance.struct_scale=1.5 \
    inference.interpolant.guidance.gamma=0.5 \
    inference.interpolant.guidance.steps=20 \
    inference.interpolant.guidance.lambda_kl=0.0

echo "✅ All experiments finished!"