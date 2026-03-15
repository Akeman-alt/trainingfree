#!/bin/bash

# 1. 定义实验子目录名称 (改成折叠任务相关的名字)
SUBDIR_NAME="run_ocflow_forward_folding_test"

echo "=========================================================="
echo "🚀 步骤 1: 开始运行 OC-Flow 引导前向折叠任务 ($SUBDIR_NAME)"
echo "=========================================================="
python -W ignore multiflow/experiments/inference_se3_flows.py \
    -cn inference_forward_folding \
    inference.interpolant.sampling.num_timesteps=100 \
    inference.interpolant.guidance.enabled=True \
    inference.interpolant.guidance.num_iters=20 \
    inference.interpolant.guidance.step_size=0.0 \
    inference.interpolant.guidance.momentum=0.0 \
    inference.inference_subdir="$SUBDIR_NAME" \
    inference.num_gpus=8

# 检查上一步是否成功执行
if [ $? -ne 0 ]; then
    echo "❌ 错误: 生成脚本运行失败，停止后续分析。"
    exit 1
fi

echo "=========================================================="
echo "📈 步骤 2: 开始生成诊断分析图表"
echo "=========================================================="

python analyze_guidance.py --subdir "$SUBDIR_NAME"

echo "=========================================================="
echo "✅ 全部任务完成！请去对应目录下查看 GLOBAL_diagnostic_report 等分析图表"
echo "=========================================================="