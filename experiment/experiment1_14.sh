#!/bin/bash

# ==========================================
# 批量实验脚本：寻找最佳的 Guidance 参数组合
# 目标：Reward > 40% 且 RMSD < 2.0
# ==========================================
NUM_GPUS=4
SAMPLES=25
# # 实验 A：稳健导航组 (High Steps, Low Gamma) - 我最看好这组
# # 逻辑：走得慢(0.1)但走得久(20步)，方向准，不崩结构
# echo "🚀 [1/6] Starting Experiment A: Gamma=0.1, Steps=20, Scale=5.0"
# python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_unconditional \
#     inference.num_gpus=$NUM_GPUS \
#     inference.inference_subdir="run_expM_scale3.0_gamma0.1_steps20_N16_kl0.01" \
#     inference.interpolant.guidance.struct_scale=3.0 \
#     inference.interpolant.guidance.gamma=0.1 \
#     inference.interpolant.guidance.steps=20 \
#     inference.interpolant.guidance.N=16 \
#     inference.interpolant.guidance.lambda_kl=0.01

# echo "🚀 [2/6] Starting Experiment A: Gamma=0.1, Steps=20, Scale=5.0"
# python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_unconditional \
#     inference.num_gpus=$NUM_GPUS \
#     inference.inference_subdir="run_expM_scale3.0_gamma0.1_steps20_N16_kl0.0" \
#     inference.interpolant.guidance.struct_scale=3.0 \
#     inference.interpolant.guidance.gamma=0.1 \
#     inference.interpolant.guidance.steps=20 \
#     inference.interpolant.guidance.N=16 \
#     inference.interpolant.guidance.lambda_kl=0.0
# echo "🚀 [3/6] Starting Experiment A: Gamma=0.1, Steps=20, Scale=5.0"
# python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_unconditional \
#     inference.num_gpus=$NUM_GPUS \
#     inference.inference_subdir="run_expM_scale3.0_gamma0.1_steps20_N16_kl0.05" \
#     inference.interpolant.guidance.struct_scale=3.0 \
#     inference.interpolant.guidance.gamma=0.1 \
#     inference.interpolant.guidance.steps=20 \
#     inference.interpolant.guidance.N=16 \
#     inference.interpolant.guidance.lambda_kl=0.05
# echo "🚀 [4/6] Starting Experiment A: Gamma=0.1, Steps=20, Scale=5.0"
# python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_unconditional \
#     inference.num_gpus=$NUM_GPUS \
#     inference.inference_subdir="run_expM_scale5.0_gamma0.1_steps20_N16_kl0.01" \
#     inference.interpolant.guidance.struct_scale=5.0 \
#     inference.interpolant.guidance.gamma=0.1 \
#     inference.interpolant.guidance.steps=20 \
#     inference.interpolant.guidance.N=16 \
#     inference.interpolant.guidance.lambda_kl=0.01

# echo "🚀 [5/6] Starting Experiment A: Gamma=0.1, Steps=20, Scale=5.0"
# python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_unconditional \
#     inference.num_gpus=$NUM_GPUS \
#     inference.inference_subdir="run_expM_scale3.0_gamma0.3_steps20_N16_kl0.01" \
#     inference.interpolant.guidance.struct_scale=3.0 \
#     inference.interpolant.guidance.gamma=0.3 \
#     inference.interpolant.guidance.steps=20 \
#     inference.interpolant.guidance.N=16 \
#     inference.interpolant.guidance.lambda_kl=0.01
# echo "🚀 [6/6] Starting Experiment A: Gamma=0.1, Steps=20, Scale=5.0"
python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_unconditional \
    inference.num_gpus=$NUM_GPUS \
    inference.inference_subdir="run_expM_scale3.0_gamma0.1_steps20_N32_kl0.01" \
    inference.interpolant.guidance.struct_scale=3.0 \
    inference.interpolant.guidance.gamma=0.1 \
    inference.interpolant.guidance.steps=20 \
    inference.interpolant.guidance.N=32 \
    inference.interpolant.guidance.lambda_kl=0.01

python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_unconditional \
    inference.num_gpus=$NUM_GPUS \
    inference.inference_subdir="run_expM_scale3.0_gamma0.1_steps20_N32_kl0.0" \
    inference.interpolant.guidance.struct_scale=3.0 \
    inference.interpolant.guidance.gamma=0.1 \
    inference.interpolant.guidance.steps=20 \
    inference.interpolant.guidance.N=32 \
    inference.interpolant.guidance.lambda_kl=0.0

python -W ignore multiflow/experiments/inference_se3_flows.py -cn inference_unconditional \
    inference.num_gpus=$NUM_GPUS \
    inference.inference_subdir="run_expM_scale3.0_gamma0.1_steps20_N128_kl0.0" \
    inference.interpolant.guidance.struct_scale=3.0 \
    inference.interpolant.guidance.gamma=0.1 \
    inference.interpolant.guidance.steps=20 \
    inference.interpolant.guidance.N=128 \
    inference.interpolant.guidance.lambda_kl=0.0
