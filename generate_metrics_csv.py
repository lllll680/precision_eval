#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从各个指标的JSON结果文件中提取数据，生成CSV统计文件

CSV格式：
- 第一行：所有数据的平均值
- 后续行：每条数据的指标值
- 最后一列：overall_acc（所有指标的平均值）
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional


def load_json_result(json_path: str) -> Optional[Dict]:
    """加载JSON结果文件"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"警告: 文件不存在 {json_path}")
        return None
    except Exception as e:
        print(f"警告: 读取文件失败 {json_path}: {e}")
        return None


def extract_metrics_from_results(
    tool_name_result: Optional[Dict],
    schema_result: Optional[Dict],
    query_param_result: Optional[Dict],
    obs_param_result: Optional[Dict],
    duplicate_result: Optional[Dict],
    state_consistency_result: Optional[Dict]
) -> List[Dict]:
    """
    从六个结果文件中提取每条数据的指标
    
    Returns:
        每条数据的指标列表，每个元素是一个字典
    """
    metrics_list = []
    
    # 构建文件路径到指标的映射
    file_metrics_map = {}
    
    # 1. 提取 tool_name_accuracy 指标
    if tool_name_result and 'per_file_results' in tool_name_result:
        for file_result in tool_name_result['per_file_results']:
            file_path = file_result['file']
            if file_path not in file_metrics_map:
                file_metrics_map[file_path] = {}
            file_metrics_map[file_path]['tool_acc'] = file_result.get('Acc_tool', 0.0)
    
    # 2. 提取 schema_validation_accuracy 指标
    if schema_result and 'per_file_results' in schema_result:
        for file_result in schema_result['per_file_results']:
            file_path = file_result['file']
            if file_path not in file_metrics_map:
                file_metrics_map[file_path] = {}
            
            # action_valid_calls 转为比例
            total_calls = file_result.get('total_calls', 0)
            action_valid = file_result.get('action_valid_calls', 0)
            obs_valid = file_result.get('observation_valid_calls', 0)
            
            file_metrics_map[file_path]['action_valid_rate'] = action_valid / total_calls if total_calls > 0 else 0.0
            file_metrics_map[file_path]['obs_valid_rate'] = obs_valid / total_calls if total_calls > 0 else 0.0
    
    # 3. 提取 query_param_accuracy 指标（如果需要的话）
    # 注意：用户没有要求这个指标，跳过
    
    # 4. 提取 obs_param_accuracy 指标
    if obs_param_result and 'per_file_results' in obs_param_result:
        for file_result in obs_param_result['per_file_results']:
            file_path = file_result['file']
            if file_path not in file_metrics_map:
                file_metrics_map[file_path] = {}
            file_metrics_map[file_path]['obs_param_acc'] = file_result.get('Acc_param_obs', 0.0)
    
    # 5. 提取 duplicate_call_rate 指标
    if duplicate_result and 'per_file_results' in duplicate_result:
        for file_result in duplicate_result['per_file_results']:
            file_path = file_result['file']
            if file_path not in file_metrics_map:
                file_metrics_map[file_path] = {}
            # 重复率转为准确率（1 - 重复率）
            file_metrics_map[file_path]['no_dup_rate'] = 1 - file_result.get('Rate_tool_dup', 0.0)
    
    # 6. 提取 state_consistency 指标
    if state_consistency_result and 'per_file_results' in state_consistency_result:
        for file_result in state_consistency_result['per_file_results']:
            file_path = file_result['file']
            if file_path not in file_metrics_map:
                file_metrics_map[file_path] = {}
            
            same_tool = file_result.get('same_tool', {})
            cross_tool = file_result.get('cross_tool', {})
            
            file_metrics_map[file_path]['same_tool_consistency'] = same_tool.get('consistency_rate', 0.0)
            file_metrics_map[file_path]['cross_tool_consistency'] = cross_tool.get('consistency_rate', 0.0)
    
    # 转换为列表，并计算 overall_acc
    for file_path, metrics in file_metrics_map.items():
        # 提取所有指标值（排除None）
        metric_values = [
            metrics.get('tool_acc'),
            metrics.get('action_valid_rate'),
            metrics.get('obs_valid_rate'),
            metrics.get('obs_param_acc'),
            metrics.get('no_dup_rate'),
            metrics.get('same_tool_consistency'),
            metrics.get('cross_tool_consistency')
        ]
        
        # 过滤掉None值
        valid_values = [v for v in metric_values if v is not None]
        
        # 计算 overall_acc（所有指标的平均值）
        overall_acc = sum(valid_values) / len(valid_values) if valid_values else 0.0
        
        metrics_list.append({
            'file_path': file_path,
            'tool_acc': metrics.get('tool_acc', 0.0),
            'action_valid_rate': metrics.get('action_valid_rate', 0.0),
            'obs_valid_rate': metrics.get('obs_valid_rate', 0.0),
            'obs_param_acc': metrics.get('obs_param_acc', 0.0),
            'no_dup_rate': metrics.get('no_dup_rate', 0.0),
            'same_tool_consistency': metrics.get('same_tool_consistency', 0.0),
            'cross_tool_consistency': metrics.get('cross_tool_consistency', 0.0),
            'overall_acc': overall_acc
        })
    
    return metrics_list


def calculate_average_metrics(metrics_list: List[Dict]) -> Dict:
    """计算所有数据的平均指标"""
    if not metrics_list:
        return {}
    
    avg_metrics = {
        'file_path': 'AVERAGE',
        'tool_acc': 0.0,
        'action_valid_rate': 0.0,
        'obs_valid_rate': 0.0,
        'obs_param_acc': 0.0,
        'no_dup_rate': 0.0,
        'same_tool_consistency': 0.0,
        'cross_tool_consistency': 0.0,
        'overall_acc': 0.0
    }
    
    n = len(metrics_list)
    
    for metrics in metrics_list:
        avg_metrics['tool_acc'] += metrics['tool_acc']
        avg_metrics['action_valid_rate'] += metrics['action_valid_rate']
        avg_metrics['obs_valid_rate'] += metrics['obs_valid_rate']
        avg_metrics['obs_param_acc'] += metrics['obs_param_acc']
        avg_metrics['no_dup_rate'] += metrics['no_dup_rate']
        avg_metrics['same_tool_consistency'] += metrics['same_tool_consistency']
        avg_metrics['cross_tool_consistency'] += metrics['cross_tool_consistency']
        avg_metrics['overall_acc'] += metrics['overall_acc']
    
    for key in avg_metrics:
        if key != 'file_path':
            avg_metrics[key] /= n
    
    return avg_metrics


def generate_metrics_csv(
    tool_name_json: str,
    schema_json: str,
    query_param_json: str,
    obs_param_json: str,
    duplicate_json: str,
    state_consistency_json: str,
    output_csv: str
):
    """
    生成指标统计CSV文件
    
    Args:
        tool_name_json: tool_name_accuracy结果文件路径
        schema_json: schema_validation_accuracy结果文件路径
        query_param_json: query_param_accuracy结果文件路径
        obs_param_json: obs_param_accuracy结果文件路径
        duplicate_json: duplicate_call_rate结果文件路径
        state_consistency_json: state_consistency结果文件路径
        output_csv: 输出CSV文件路径
    """
    print("开始生成指标统计CSV...")
    
    # 加载所有JSON结果文件
    print("\n加载JSON结果文件...")
    tool_name_result = load_json_result(tool_name_json)
    schema_result = load_json_result(schema_json)
    query_param_result = load_json_result(query_param_json)
    obs_param_result = load_json_result(obs_param_json)
    duplicate_result = load_json_result(duplicate_json)
    state_consistency_result = load_json_result(state_consistency_json)
    
    # 提取指标
    print("\n提取指标...")
    metrics_list = extract_metrics_from_results(
        tool_name_result,
        schema_result,
        query_param_result,
        obs_param_result,
        duplicate_result,
        state_consistency_result
    )
    
    if not metrics_list:
        print("错误: 没有提取到任何指标数据")
        return
    
    print(f"提取到 {len(metrics_list)} 条数据的指标")
    
    # 计算平均值
    print("\n计算平均指标...")
    avg_metrics = calculate_average_metrics(metrics_list)
    
    # 写入CSV文件
    print(f"\n生成CSV文件: {output_csv}")
    
    # 定义列名（更规范的命名）
    fieldnames = [
        'file_path',
        'tool_acc',
        'action_valid_rate',
        'obs_valid_rate',
        'obs_param_acc',
        'no_dup_rate',
        'same_tool_consistency',
        'cross_tool_consistency',
        'overall_acc'
    ]
    
    # 定义列的显示名称
    header_display = {
        'file_path': 'File Path',
        'tool_acc': 'Tool Accuracy',
        'action_valid_rate': 'Action Valid Rate',
        'obs_valid_rate': 'Obs Valid Rate',
        'obs_param_acc': 'Obs Param Accuracy',
        'no_dup_rate': 'No Duplicate Rate',
        'same_tool_consistency': 'Same Tool Consistency',
        'cross_tool_consistency': 'Cross Tool Consistency',
        'overall_acc': 'Overall Accuracy'
    }
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # 写入表头
        writer.writerow(header_display)
        
        # 写入平均值行（第一行数据）
        writer.writerow(avg_metrics)
        
        # 写入每条数据的指标
        for metrics in sorted(metrics_list, key=lambda x: x['file_path']):
            writer.writerow(metrics)
    
    print(f"\n✅ CSV文件生成成功: {output_csv}")
    print(f"\n平均指标:")
    print(f"  Tool Accuracy: {avg_metrics['tool_acc']:.4f} ({avg_metrics['tool_acc']*100:.2f}%)")
    print(f"  Action Valid Rate: {avg_metrics['action_valid_rate']:.4f} ({avg_metrics['action_valid_rate']*100:.2f}%)")
    print(f"  Obs Valid Rate: {avg_metrics['obs_valid_rate']:.4f} ({avg_metrics['obs_valid_rate']*100:.2f}%)")
    print(f"  Obs Param Accuracy: {avg_metrics['obs_param_acc']:.4f} ({avg_metrics['obs_param_acc']*100:.2f}%)")
    print(f"  No Duplicate Rate: {avg_metrics['no_dup_rate']:.4f} ({avg_metrics['no_dup_rate']*100:.2f}%)")
    print(f"  Same Tool Consistency: {avg_metrics['same_tool_consistency']:.4f} ({avg_metrics['same_tool_consistency']*100:.2f}%)")
    print(f"  Cross Tool Consistency: {avg_metrics['cross_tool_consistency']:.4f} ({avg_metrics['cross_tool_consistency']*100:.2f}%)")
    print(f"  Overall Accuracy: {avg_metrics['overall_acc']:.4f} ({avg_metrics['overall_acc']*100:.2f}%)")


if __name__ == "__main__":
    # 配置JSON结果文件路径
    tool_name_json = "tool_name_accuracy_result.json"
    schema_json = "schema_validation_accuracy_result.json"
    query_param_json = "query_param_accuracy_result.json"
    obs_param_json = "obs_param_accuracy_result.json"
    duplicate_json = "duplicate_call_rate_result.json"
    state_consistency_json = "state_consistency_result.json"
    
    # 输出CSV文件路径
    output_csv = "metrics_summary.csv"
    
    # 生成CSV
    generate_metrics_csv(
        tool_name_json,
        schema_json,
        query_param_json,
        obs_param_json,
        duplicate_json,
        state_consistency_json,
        output_csv
    )
