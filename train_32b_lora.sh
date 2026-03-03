#!/bin/bash
# -*- coding: utf-8 -*-
# 32B模型 LoRA微调脚本 - 4张A100 GPU

set -e  # 遇到错误立即退出

echo "=========================================="
echo "开始训练 32B 模型 LoRA 微调"
echo "=========================================="
echo "GPU: 4张A100"
echo "数据集: training_data.jsonl"
echo "输出目录: output"
echo "=========================================="
echo ""

CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model Qwen/Qwen2.5-32B-Instruct \
    --train_type lora \
    --tuner_backend peft \
    --dataset /Users/liaoying/Desktop/研一/llm/data_eval/precision_index/training_data.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-5 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing true \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --max_length 4096 \
    --output_dir /Users/liaoying/Desktop/研一/llm/data_eval/precision_index/output \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --dataloader_num_workers 4 \
    --deepspeed default-zero2 \
    --save_only_model true \
    --use_flash_attn true

echo ""
echo "=========================================="
echo "✅ 训练完成！"
echo "=========================================="
echo "模型保存在: output/"
echo ""
