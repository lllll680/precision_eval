#!/bin/bash
# -*- coding: utf-8 -*-
#
# 自动运行所有指标评估脚本并生成CSV统计文件
#
# 使用方法:
#   chmod +x run_all_metrics.sh
#   ./run_all_metrics.sh
#

set -e  # 遇到错误立即退出

echo "=========================================="
echo "开始运行所有指标评估脚本"
echo "=========================================="
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 1. 运行工具名正确率评估
echo "[1/6] 运行工具名正确率评估 (tool_name_accuracy.py)..."
python tool_name_accuracy.py
if [ $? -eq 0 ]; then
    echo "✅ 工具名正确率评估完成"
else
    echo "❌ 工具名正确率评估失败"
    exit 1
fi
echo ""

# 2. 运行Schema合法率评估
echo "[2/6] 运行Schema合法率评估 (schema_validation_accuracy.py)..."
python schema_validation_accuracy.py
if [ $? -eq 0 ]; then
    echo "✅ Schema合法率评估完成"
else
    echo "❌ Schema合法率评估失败"
    exit 1
fi
echo ""

# 3. 运行Query参数引用正确率评估
echo "[3/6] 运行Query参数引用正确率评估 (query_param_accuracy.py)..."
python query_param_accuracy.py
if [ $? -eq 0 ]; then
    echo "✅ Query参数引用正确率评估完成"
else
    echo "❌ Query参数引用正确率评估失败"
    exit 1
fi
echo ""

# 4. 运行Observation参数引用正确率评估
echo "[4/6] 运行Observation参数引用正确率评估 (obs_param_accuracy.py)..."
python obs_param_accuracy.py
if [ $? -eq 0 ]; then
    echo "✅ Observation参数引用正确率评估完成"
else
    echo "❌ Observation参数引用正确率评估失败"
    exit 1
fi
echo ""

# 5. 运行重复调用率评估
echo "[5/6] 运行重复调用率评估 (duplicate_call_rate.py)..."
python duplicate_call_rate.py
if [ $? -eq 0 ]; then
    echo "✅ 重复调用率评估完成"
else
    echo "❌ 重复调用率评估失败"
    exit 1
fi
echo ""

# 6. 运行状态反馈一致率评估
echo "[6/6] 运行状态反馈一致率评估 (state_consistency.py)..."
python state_consistency.py
if [ $? -eq 0 ]; then
    echo "✅ 状态反馈一致率评估完成"
else
    echo "❌ 状态反馈一致率评估失败"
    exit 1
fi
echo ""

# 7. 生成CSV统计文件
echo "=========================================="
echo "生成CSV统计文件 (generate_metrics_csv.py)..."
echo "=========================================="
python generate_metrics_csv.py
if [ $? -eq 0 ]; then
    echo "✅ CSV统计文件生成完成"
else
    echo "❌ CSV统计文件生成失败"
    exit 1
fi
echo ""

# 计算总耗时
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED_TIME / 60))
SECONDS=$((ELAPSED_TIME % 60))

echo "=========================================="
echo "✅ 所有指标评估完成！"
echo "=========================================="
echo ""
echo "生成的文件:"
echo "  - tool_name_accuracy_result.json"
echo "  - schema_validation_accuracy_result.json"
echo "  - query_param_accuracy_result.json"
echo "  - obs_param_accuracy_result.json"
echo "  - duplicate_call_rate_result.json"
echo "  - state_consistency_result.json"
echo "  - metrics_summary.csv"
echo ""
echo "总耗时: ${MINUTES}分${SECONDS}秒"
echo ""


tmux new-session -s metrics "bash /Users/liaoying/Desktop/研一/llm/data_eval/precision_index/run_all_metrics.sh 2>&1 | tee /Users/liaoying/Desktop/研一/llm/data_eval/precision_index/run_metrics_log.txt"
