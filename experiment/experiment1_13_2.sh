#!/bin/bash

# ==========================================
# 终局之战：联合流形引导 (Joint Manifold Guidance) 最终参数验证
# 目标：锁定 SOTA 结果 (Reward > 24%, RMSD < 1.8 Å)
# ==========================================

# 通用设置
NUM_GPUS=4
SAMPLES=25  # 4卡并行共100个样本，保证统计显著性

# ------------------------------------------------------------------
# 实验 D: 终局微调 (Fine-tuning) - 最强候选
# 配置: Scale=4.0, Gamma=0.1, Steps=20, No KL
# 预期: RMSD 完美回落到 1.8 Å 以内，Reward 保持高位
# ------------------------------------------------------------------
echo "🚀 [1/2] Starting Experiment D: The Final Polish (Scale=4.0)..."
python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_unconditional \
    inference.num_gpus=$NUM_GPUS \
    inference.samples.samples_per_length=$SAMPLES \
    inference.inference_subdir="run_expD_scale4.0_gamma0.1_steps20" \
    inference.interpolant.guidance.struct_scale=4.0 \
    inference.interpolant.guidance.gamma=0.1 \
    inference.interpolant.guidance.steps=20 \
    inference.interpolant.guidance.lambda_kl=0.0

# ------------------------------------------------------------------
# 实验 E: 对照组 (KL Regularization) - 探索约束边界
# 配置: Scale=5.0, Gamma=0.1, Steps=20, KL=0.05
# 预期: 用 KL 惩罚代替物理降力。观察 KL 是否能作为保护结构的另一种手段。
# ------------------------------------------------------------------
echo "🚀 [2/2] Starting Experiment E: KL Control (Scale=5.0 + KL=0.05)..."
python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_unconditional \
    inference.num_gpus=$NUM_GPUS \
    inference.samples.samples_per_length=$SAMPLES \
    inference.inference_subdir="run_expE_scale5.0_gamma0.1_steps20_kl0.05" \
    inference.interpolant.guidance.struct_scale=5.0 \
    inference.interpolant.guidance.gamma=0.1 \
    inference.interpolant.guidance.steps=20 \
    inference.interpolant.guidance.lambda_kl=0.05

echo "✅ All final experiments finished! Ready for plotting."