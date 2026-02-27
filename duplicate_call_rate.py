#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""重复调用率计算脚本

指标定义：
1. Rate_tool_dup: 工具重复率（工具名相同即算重复）
2. Rate_consecutive_dup: 连续重复率（与上一次调用完全相同，反映模型卡住）

说明：
- 每个JSON文件是一条独立对话
- 文件级指标：评估单条对话内的重复情况
- Overall指标：所有对话的平均重复率
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple, Optional
from collections import defaultdict


def normalize_args(args: Dict) -> str:
    """
    将参数字典标准化为可哈希的字符串
    
    Args:
        args: 参数字典
        
    Returns:
        标准化后的字符串（JSON格式，键排序）
    """
    if not args or not isinstance(args, dict):
        return "{}"
    
    # 递归处理嵌套结构，确保一致性
    def normalize_value(v):
        if isinstance(v, dict):
            return {k: normalize_value(v2) for k, v2 in sorted(v.items())}
        elif isinstance(v, list):
            return [normalize_value(item) for item in v]
        elif v is None:
            return None
        else:
            return str(v)
    
    normalized = {k: normalize_value(v) for k, v in sorted(args.items())}
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False)


def get_call_signature(tool_name: str, args: Dict) -> str:
    """
    生成调用签名：tool_name + normalized_args
    
    Args:
        tool_name: 工具名称
        args: 参数字典
        
    Returns:
        调用签名字符串
    """
    return f"{tool_name}::{normalize_args(args)}"


def calculate_param_similarity(args1: Dict, args2: Dict) -> float:
    """
    计算两个参数字典的相似度
    
    Args:
        args1: 第一个参数字典
        args2: 第二个参数字典
        
    Returns:
        相似度 (0-1)，相同参数数 / 总参数数
    """
    if not args1 and not args2:
        return 1.0
    
    if not args1 or not args2:
        return 0.0
    
    # 获取所有参数名
    all_keys = set(args1.keys()) | set(args2.keys())
    
    if not all_keys:
        return 1.0
    
    # 计算相同参数数
    same_count = 0
    for key in all_keys:
        if key in args1 and key in args2:
            # 比较值是否相同（转换为字符串比较）
            if str(args1[key]) == str(args2[key]):
                same_count += 1
    
    return same_count / len(all_keys)


def extract_all_calls(response: List, exclude_last_steps: int = 0) -> List[Dict]:
    """
    从 response 中提取所有工具调用
    
    Args:
        response: response 列表
        exclude_last_steps: 排除最后 N 个 steps
        
    Returns:
        调用列表，每个元素为 {'tool_name': str, 'args': dict, 'step': str}
    """
    calls = []
    
    # 计算需要处理的 steps 数量
    total_steps = len(response)
    steps_to_process = max(0, total_steps - exclude_last_steps)
    
    for step_index, step_item in enumerate(response[:steps_to_process]):
        for step_key, step_data in step_item.items():
            if not step_key.startswith('step'):
                continue
            
            if not isinstance(step_data, dict) or 'coa' not in step_data:
                continue
            
            for coa_item in step_data['coa']:
                action = coa_item.get('action', {})
                if not isinstance(action, dict):
                    continue
                
                tool_name = action.get('name', '')
                args = action.get('args', {})
                
                if tool_name:
                    calls.append({
                        'tool_name': tool_name,
                        'args': args if isinstance(args, dict) else {},
                        'step': step_key
                    })
    
    return calls


def calculate_duplicate_metrics(calls: List[Dict]) -> Dict:
    """
    计算重复调用相关的各项指标
    
    Args:
        calls: 调用列表
        
    Returns:
        包含各项指标的字典
    """
    total_calls = len(calls)
    
    if total_calls == 0:
        return {
            'total_calls': 0,
            'tool_dup_count': 0,
            'consecutive_dup_count': 0,
            'Rate_tool_dup': 0.0,
            'Rate_consecutive_dup': 0.0,
            'duplicate_details': []
        }
    
    # 用于跟踪已出现的工具名
    seen_tools = set()
    
    # 统计变量
    tool_dup_count = 0
    consecutive_dup_count = 0
    
    # 按工具名分组，记录每个工具的所有step
    tool_steps_map = defaultdict(list)
    
    # 上一次调用的签名（用于检测连续重复）
    prev_signature = None
    
    for call in calls:
        tool_name = call['tool_name']
        args = call['args']
        step = call['step']
        signature = get_call_signature(tool_name, args)
        
        # 记录该工具的step
        tool_steps_map[tool_name].append(step)
        
        # 检查工具名重复
        if tool_name in seen_tools:
            tool_dup_count += 1
        seen_tools.add(tool_name)
        
        # 检查连续重复（工具名+参数完全相同）
        if prev_signature is not None and signature == prev_signature:
            consecutive_dup_count += 1
        prev_signature = signature
    
    # 构建duplicate_details：只包含调用次数>1的工具
    duplicate_details = [
        {
            'tool_name': tool_name,
            'steps': steps,
            'call_count': len(steps)
        }
        for tool_name, steps in sorted(tool_steps_map.items())
        if len(steps) > 1
    ]
    
    return {
        'total_calls': total_calls,
        'tool_dup_count': tool_dup_count,
        'consecutive_dup_count': consecutive_dup_count,
        'Rate_tool_dup': tool_dup_count / total_calls,
        'Rate_consecutive_dup': consecutive_dup_count / total_calls,
        'duplicate_details': duplicate_details
    }


def calculate_duplicate_call_rate(
    data_folders: List[str],
    exclude_last_steps: int = 0
) -> Dict:
    """
    计算重复调用率
    
    Args:
        data_folders: 数据文件夹路径列表
        exclude_last_steps: 排除最后 N 个 steps（默认 0 不排除，设为 2 排除最后两个总结 step）
        
    Returns:
        包含统计结果的字典
    """
    per_file_results = []
    
    # 遍历数据文件夹
    for folder in data_folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"警告: 文件夹不存在 {folder}")
            continue
        
        print(f"\n处理数据文件夹: {folder}")
        
        for json_file in sorted(folder_path.glob("*.json")):
            # 跳过特殊文件
            if json_file.name in ['question_info.json', 'batch_summary.json']:
                continue
            
            print(f"  处理文件: {json_file.name}")
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                response = data.get('response', [])
                
                # 提取所有调用
                calls = extract_all_calls(response, exclude_last_steps)
                
                if exclude_last_steps > 0:
                    total_steps = len(response)
                    steps_processed = max(0, total_steps - exclude_last_steps)
                    print(f"    排除最后 {exclude_last_steps} 个 steps（共 {total_steps} 个，处理前 {steps_processed} 个）")
                
                # 计算该文件的重复指标
                metrics = calculate_duplicate_metrics(calls)
                
                # 记录文件结果
                file_result = {
                    'file': str(json_file),
                    'total_calls': metrics['total_calls'],
                    'tool_dup_count': metrics['tool_dup_count'],
                    'consecutive_dup_count': metrics['consecutive_dup_count'],
                    'Rate_tool_dup': metrics['Rate_tool_dup'],
                    'Rate_consecutive_dup': metrics['Rate_consecutive_dup'],
                    'duplicate_details': metrics['duplicate_details']
                }
                per_file_results.append(file_result)
                
            except Exception as e:
                print(f"    处理文件 {json_file} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 计算总体指标（文件内平均）
    if per_file_results:
        avg_rate_tool_dup = sum(f['Rate_tool_dup'] for f in per_file_results) / len(per_file_results)
        avg_rate_consecutive_dup = sum(f['Rate_consecutive_dup'] for f in per_file_results) / len(per_file_results)
    else:
        avg_rate_tool_dup = 0.0
        avg_rate_consecutive_dup = 0.0
    
    overall = {
        'total_files': len(per_file_results),
        'avg_rate_tool_dup': avg_rate_tool_dup,
        'avg_rate_consecutive_dup': avg_rate_consecutive_dup
    }
    
    return {
        'per_file_results': per_file_results,
        'overall': overall,
        'config': {
            'exclude_last_steps': exclude_last_steps
        }
    }


def print_results(result: Dict):
    """打印结果"""
    print("\n" + "="*60)
    print("重复调用率统计结果")
    print("="*60)
    
    config = result.get('config', {})
    print(f"\n配置: 排除最后 {config.get('exclude_last_steps', 0)} 个 steps")
    
    # 打印每条数据的结果
    print("\n每条数据的结果:")
    print("-"*60)
    for file_result in result['per_file_results']:
        print(f"\n文件: {file_result['file']}")
        print(f"  总调用数: {file_result['total_calls']}")
        print(f"  工具重复数: {file_result['tool_dup_count']}")
        print(f"  连续重复数: {file_result['consecutive_dup_count']}")
        print(f"  Rate_tool_dup: {file_result['Rate_tool_dup']:.4f} ({file_result['Rate_tool_dup']*100:.2f}%)")
        print(f"  Rate_consecutive_dup: {file_result['Rate_consecutive_dup']:.4f} ({file_result['Rate_consecutive_dup']*100:.2f}%)")
        
        if file_result['duplicate_details']:
            print(f"  重复工具详情:")
            for detail in file_result['duplicate_details'][:5]:  # 只显示前5个
                steps_str = ', '.join(detail['steps'])
                print(f"    - {detail['tool_name']} (调用{detail['call_count']}次): {steps_str}")
            if len(file_result['duplicate_details']) > 5:
                print(f"    ... 还有 {len(file_result['duplicate_details']) - 5} 个重复工具")
    
    # 打印总体结果
    print("\n" + "="*60)
    print("总体统计结果（文件内平均）")
    print("="*60)
    overall = result['overall']
    print(f"总文件数: {overall['total_files']}")
    print(f"\n平均工具重复率 (avg_rate_tool_dup): {overall['avg_rate_tool_dup']:.4f} ({overall['avg_rate_tool_dup']*100:.2f}%)")
    print(f"平均连续重复率 (avg_rate_consecutive_dup): {overall['avg_rate_consecutive_dup']:.4f} ({overall['avg_rate_consecutive_dup']*100:.2f}%)")
    print("="*60)


if __name__ == "__main__":
    # 配置路径
    data_folders = [
        "/mnt/data/kw/ly/precision_index/data1",
        "/mnt/data/kw/ly/precision_index/data2",
    ]
    
    # 排除最后 N 个 steps（设为 2 排除最后两个总结 step，设为 0 不排除）
    exclude_last_steps = 0
    
    # 计算重复调用率
    result = calculate_duplicate_call_rate(data_folders, exclude_last_steps)
    
    # 打印结果
    print_results(result)
    
    # 保存结果到JSON文件
    output_file = "duplicate_call_rate_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_file}")
