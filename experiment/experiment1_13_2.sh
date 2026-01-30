#!/bin/bash

# ==========================================
# 终极突破：高精度采样 (High N) 微调实验
# 目标：利用 N=16 的高 Reward 潜力，配合低 Scale 压住 RMSD
# 预期：Reward > 30% 且 RMSD < 1.9 Å
# ==========================================

NUM_GPUS=4
SAMPLES=25

# ------------------------------------------------------------------
# 实验 G (黄金分割点): Scale=3.0 + N=16
# 逻辑: N=16 带来了 36% 的 Reward，Scale 5->3 (降40%) 应该能把 RMSD 从 2.6 压回 1.8 左右
# ------------------------------------------------------------------
echo "🚀 [1/2] Starting Experiment G: High Precision (N=16, Scale=3.0)..."
python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_unconditional \
    inference.num_gpus=$NUM_GPUS \
    inference.samples.samples_per_length=$SAMPLES \
    inference.inference_subdir="run_expH_scale3.0_gamma0.1_steps20_N16" \
    inference.interpolant.guidance.struct_scale=3.0 \
    inference.interpolant.guidance.gamma=0.1 \
    inference.interpolant.guidance.steps=20 \
    inference.interpolant.guidance.N=16 \
    inference.interpolant.guidance.lambda_kl=0.0

# ------------------------------------------------------------------
# 实验 H (极度稳健): Scale=2.5 + N=16
# 逻辑: 如果 Scale=3.0 还是压不住，2.5 绝对安全。
# ------------------------------------------------------------------
echo "🚀 [2/2] Starting Experiment H: Conservative Precision (N=16, Scale=2.5)..."
python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_unconditional \
    inference.num_gpus=$NUM_GPUS \
    inference.samples.samples_per_length=$SAMPLES \
    inference.inference_subdir="run_expI_scale2_gamma0.1_steps20_N16" \
    inference.interpolant.guidance.struct_scale=2 \
    inference.interpolant.guidance.gamma=0.1 \
    inference.interpolant.guidance.steps=20 \
    inference.interpolant.guidance.N=16 \
    inference.interpolant.guidance.lambda_kl=0.0


python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_unconditional \
    inference.num_gpus=$NUM_GPUS \
    inference.samples.samples_per_length=$SAMPLES \
    inference.inference_subdir="run_expJ_scale3.0_gamma0.1_steps20_N32" \
    inference.interpolant.guidance.struct_scale=3.0 \
    inference.interpolant.guidance.gamma=0.1 \
    inference.interpolant.guidance.steps=20 \
    inference.interpolant.guidance.N=32 \
    inference.interpolant.guidance.lambda_kl=0.0



python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_unconditional \
    inference.num_gpus=$NUM_GPUS \
    inference.samples.samples_per_length=$SAMPLES \
    inference.inference_subdir="run_expK_scale3.0_gamma0.1_steps20_N32" \
    inference.interpolant.guidance.struct_scale=3.0 \
    inference.interpolant.guidance.gamma=0.1 \
    inference.interpolant.guidance.steps=20 \
    inference.interpolant.guidance.N=32 \
    inference.interpolant.guidance.lambda_kl=0.0


python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_unconditional \
    inference.num_gpus=$NUM_GPUS \
    inference.samples.samples_per_length=$SAMPLES \
    inference.inference_subdir="run_expL_scale2.0_gamma0.1_steps20_N16_kl0.01" \
    inference.interpolant.guidance.struct_scale=2.0 \
    inference.interpolant.guidance.gamma=0.1 \
    inference.interpolant.guidance.steps=20 \
    inference.interpolant.guidance.N=16 \
    inference.interpolant.guidance.lambda_kl=0.01



python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_unconditional \
    inference.num_gpus=$NUM_GPUS \
    inference.samples.samples_per_length=$SAMPLES \
    inference.inference_subdir="run_expM_scale3.0_gamma0.1_steps20_N16_kl0.01" \
    inference.interpolant.guidance.struct_scale=3.0 \
    inference.interpolant.guidance.gamma=0.1 \
    inference.interpolant.guidance.steps=20 \
    inference.interpolant.guidance.N=16 \
    inference.interpolant.guidance.lambda_kl=0.01

echo "✅ Optimization finished. Check the Pareto Frontier!"