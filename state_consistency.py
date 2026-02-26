#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
状态反馈一致率计算脚本

指标定义：
1. 同工具一致率 (SameToolConsistency): 同一工具多次调用时，相同属性的值是否一致
2. 跨工具一致率 (CrossToolConsistency): 不同工具返回相同属性名时，值是否一致

假设：所有工具都是只读查询操作，任何值变化都记为冲突
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple, Optional
from collections import defaultdict


def normalize_value(value: Any) -> str:
    """
    将值标准化为可比较的字符串
    
    Args:
        value: 任意类型的值
        
    Returns:
        标准化后的字符串
    """
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        return value
    elif isinstance(value, list):
        # 对列表元素排序后转字符串（忽略顺序）
        normalized_items = sorted([normalize_value(item) for item in value])
        return json.dumps(normalized_items, ensure_ascii=False, sort_keys=True)
    elif isinstance(value, dict):
        # 对字典键排序后转字符串
        normalized_dict = {k: normalize_value(v) for k, v in sorted(value.items())}
        return json.dumps(normalized_dict, ensure_ascii=False, sort_keys=True)
    else:
        return str(value)


def flatten_observation(observation: Any, prefix: str = "", max_depth: int = 3) -> Dict[str, str]:
    """
    将 observation 展开为扁平的属性-值对
    
    Args:
        observation: observation 数据
        prefix: 当前路径前缀
        max_depth: 最大展开深度
        
    Returns:
        扁平化的 {属性路径: 标准化值} 字典
    """
    result = {}
    
    if max_depth <= 0:
        if prefix:
            result[prefix] = normalize_value(observation)
        return result
    
    if isinstance(observation, dict):
        for key, value in observation.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                result.update(flatten_observation(value, new_key, max_depth - 1))
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                # 数组中的对象：不展开索引，只记录数组整体
                result[new_key] = normalize_value(value)
            else:
                result[new_key] = normalize_value(value)
    elif isinstance(observation, list):
        # 顶层是数组的情况
        result[prefix if prefix else "root"] = normalize_value(observation)
    else:
        result[prefix if prefix else "value"] = normalize_value(observation)
    
    return result


def extract_observations_from_response(response: List, exclude_last_steps: int = 0) -> List[Dict]:
    """
    从 response 中提取所有 observation 记录
    
    Args:
        response: response 列表
        exclude_last_steps: 排除最后 N 个 steps
        
    Returns:
        观测记录列表，每个元素为 {
            'tool_name': str,
            'step': str,
            'step_index': int,
            'coa_index': int,
            'attributes': {属性名: 值}
        }
    """
    records = []
    
    total_steps = len(response)
    steps_to_process = max(0, total_steps - exclude_last_steps)
    
    for step_index, step_item in enumerate(response[:steps_to_process]):
        for step_key, step_data in step_item.items():
            if not step_key.startswith('step'):
                continue
            
            if not isinstance(step_data, dict) or 'coa' not in step_data:
                continue
            
            for coa_index, coa_item in enumerate(step_data['coa']):
                action = coa_item.get('action', {})
                observation = coa_item.get('observation')
                
                if not isinstance(action, dict) or observation is None:
                    continue
                
                tool_name = action.get('name', '')
                if not tool_name:
                    continue
                
                # 展开 observation 为扁平属性
                attributes = flatten_observation(observation)
                
                records.append({
                    'tool_name': tool_name,
                    'step': step_key,
                    'step_index': step_index,
                    'coa_index': coa_index,
                    'attributes': attributes
                })
    
    return records


def calculate_same_tool_consistency(records: List[Dict]) -> Dict:
    """
    计算同工具一致率
    
    按 (tool_name, attribute) 分组，检查同一工具多次调用时属性值是否一致
    
    Args:
        records: 观测记录列表
        
    Returns:
        {
            'total_pairs': int,  # 总比较对数
            'conflict_pairs': int,  # 冲突对数
            'consistency_rate': float,  # 一致率
            'conflicts': [...]  # 冲突详情
        }
    """
    # 按 (tool_name, attribute) 分组
    tool_attr_values = defaultdict(list)
    
    for record in records:
        tool_name = record['tool_name']
        for attr, value in record['attributes'].items():
            key = (tool_name, attr)
            tool_attr_values[key].append({
                'value': value,
                'step': record['step'],
                'step_index': record['step_index'],
                'coa_index': record['coa_index']
            })
    
    total_pairs = 0
    conflict_pairs = 0
    conflicts = []
    
    for (tool_name, attr), value_records in tool_attr_values.items():
        if len(value_records) < 2:
            continue
        
        # 比较所有对
        for i in range(len(value_records) - 1):
            for j in range(i + 1, len(value_records)):
                total_pairs += 1
                
                if value_records[i]['value'] != value_records[j]['value']:
                    conflict_pairs += 1
                    conflicts.append({
                        'tool_name': tool_name,
                        'attribute': attr,
                        'observation_1': {
                            'step': value_records[i]['step'],
                            'value': value_records[i]['value']
                        },
                        'observation_2': {
                            'step': value_records[j]['step'],
                            'value': value_records[j]['value']
                        }
                    })
    
    consistency_rate = 1 - (conflict_pairs / total_pairs) if total_pairs > 0 else 1.0
    
    return {
        'total_pairs': total_pairs,
        'conflict_pairs': conflict_pairs,
        'consistency_rate': consistency_rate,
        'conflicts': conflicts
    }


def calculate_cross_tool_consistency(records: List[Dict]) -> Dict:
    """
    计算跨工具一致率
    
    按 attribute 分组（忽略 tool_name），检查不同工具返回相同属性名时值是否一致
    
    Args:
        records: 观测记录列表
        
    Returns:
        {
            'total_pairs': int,  # 总比较对数
            'conflict_pairs': int,  # 冲突对数
            'consistency_rate': float,  # 一致率
            'conflicts': [...]  # 冲突详情
        }
    """
    # 按 attribute 分组
    attr_values = defaultdict(list)
    
    for record in records:
        tool_name = record['tool_name']
        for attr, value in record['attributes'].items():
            attr_values[attr].append({
                'value': value,
                'tool_name': tool_name,
                'step': record['step'],
                'step_index': record['step_index'],
                'coa_index': record['coa_index']
            })
    
    total_pairs = 0
    conflict_pairs = 0
    conflicts = []
    
    for attr, value_records in attr_values.items():
        if len(value_records) < 2:
            continue
        
        # 只比较不同工具之间的值
        for i in range(len(value_records) - 1):
            for j in range(i + 1, len(value_records)):
                # 跳过同一工具的比较（已在 same_tool 中计算）
                if value_records[i]['tool_name'] == value_records[j]['tool_name']:
                    continue
                
                total_pairs += 1
                
                if value_records[i]['value'] != value_records[j]['value']:
                    conflict_pairs += 1
                    conflicts.append({
                        'attribute': attr,
                        'observation_1': {
                            'tool_name': value_records[i]['tool_name'],
                            'step': value_records[i]['step'],
                            'value': value_records[i]['value']
                        },
                        'observation_2': {
                            'tool_name': value_records[j]['tool_name'],
                            'step': value_records[j]['step'],
                            'value': value_records[j]['value']
                        }
                    })
    
    consistency_rate = 1 - (conflict_pairs / total_pairs) if total_pairs > 0 else 1.0
    
    return {
        'total_pairs': total_pairs,
        'conflict_pairs': conflict_pairs,
        'consistency_rate': consistency_rate,
        'conflicts': conflicts
    }


def calculate_state_consistency(
    data_folders: List[str],
    exclude_last_steps: int = 0
) -> Dict:
    """
    计算状态反馈一致率
    
    Args:
        data_folders: 数据文件夹路径列表
        exclude_last_steps: 排除最后 N 个 steps
        
    Returns:
        包含统计结果的字典
    """
    per_file_results = []
    
    # 总体统计
    same_tool_total_pairs = 0
    same_tool_conflict_pairs = 0
    cross_tool_total_pairs = 0
    cross_tool_conflict_pairs = 0
    
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
                
                # 提取所有 observation 记录
                records = extract_observations_from_response(response, exclude_last_steps)
                
                if exclude_last_steps > 0:
                    total_steps = len(response)
                    steps_processed = max(0, total_steps - exclude_last_steps)
                    print(f"    排除最后 {exclude_last_steps} 个 steps（共 {total_steps} 个，处理前 {steps_processed} 个）")
                
                # 计算同工具一致率
                same_tool_result = calculate_same_tool_consistency(records)
                
                # 计算跨工具一致率
                cross_tool_result = calculate_cross_tool_consistency(records)
                
                # 累加到总体统计
                same_tool_total_pairs += same_tool_result['total_pairs']
                same_tool_conflict_pairs += same_tool_result['conflict_pairs']
                cross_tool_total_pairs += cross_tool_result['total_pairs']
                cross_tool_conflict_pairs += cross_tool_result['conflict_pairs']
                
                # 记录文件结果
                file_result = {
                    'file': str(json_file),
                    'observation_count': len(records),
                    'same_tool': {
                        'total_pairs': same_tool_result['total_pairs'],
                        'conflict_pairs': same_tool_result['conflict_pairs'],
                        'consistency_rate': same_tool_result['consistency_rate'],
                        'conflicts': same_tool_result['conflicts'][:10]  # 只保留前10个
                    },
                    'cross_tool': {
                        'total_pairs': cross_tool_result['total_pairs'],
                        'conflict_pairs': cross_tool_result['conflict_pairs'],
                        'consistency_rate': cross_tool_result['consistency_rate'],
                        'conflicts': cross_tool_result['conflicts'][:10]
                    }
                }
                per_file_results.append(file_result)
                
            except Exception as e:
                print(f"    处理文件 {json_file} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 计算总体指标
    overall = {
        'same_tool': {
            'total_pairs': same_tool_total_pairs,
            'conflict_pairs': same_tool_conflict_pairs,
            'consistency_rate': 1 - (same_tool_conflict_pairs / same_tool_total_pairs) if same_tool_total_pairs > 0 else 1.0
        },
        'cross_tool': {
            'total_pairs': cross_tool_total_pairs,
            'conflict_pairs': cross_tool_conflict_pairs,
            'consistency_rate': 1 - (cross_tool_conflict_pairs / cross_tool_total_pairs) if cross_tool_total_pairs > 0 else 1.0
        }
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
    print("状态反馈一致率统计结果")
    print("="*60)
    
    config = result.get('config', {})
    print(f"\n配置: 排除最后 {config.get('exclude_last_steps', 0)} 个 steps")
    
    # 打印每条数据的结果
    print("\n每条数据的结果:")
    print("-"*60)
    for file_result in result['per_file_results']:
        print(f"\n文件: {file_result['file']}")
        print(f"  观测记录数: {file_result['observation_count']}")
        
        same_tool = file_result['same_tool']
        print(f"  同工具一致率:")
        print(f"    比较对数: {same_tool['total_pairs']}")
        print(f"    冲突对数: {same_tool['conflict_pairs']}")
        print(f"    一致率: {same_tool['consistency_rate']:.4f} ({same_tool['consistency_rate']*100:.2f}%)")
        
        if same_tool['conflicts']:
            print(f"    冲突详情 (前{min(3, len(same_tool['conflicts']))}条):")
            for conflict in same_tool['conflicts'][:3]:
                val1 = str(conflict['observation_1']['value'])[:50]
                val2 = str(conflict['observation_2']['value'])[:50]
                print(f"      - [{conflict['tool_name']}] {conflict['attribute']}")
                print(f"        {conflict['observation_1']['step']}: {val1}...")
                print(f"        {conflict['observation_2']['step']}: {val2}...")
        
        cross_tool = file_result['cross_tool']
        print(f"  跨工具一致率:")
        print(f"    比较对数: {cross_tool['total_pairs']}")
        print(f"    冲突对数: {cross_tool['conflict_pairs']}")
        print(f"    一致率: {cross_tool['consistency_rate']:.4f} ({cross_tool['consistency_rate']*100:.2f}%)")
        
        if cross_tool['conflicts']:
            print(f"    冲突详情 (前{min(3, len(cross_tool['conflicts']))}条):")
            for conflict in cross_tool['conflicts'][:3]:
                val1 = str(conflict['observation_1']['value'])[:40]
                val2 = str(conflict['observation_2']['value'])[:40]
                print(f"      - 属性: {conflict['attribute']}")
                print(f"        {conflict['observation_1']['tool_name']}@{conflict['observation_1']['step']}: {val1}...")
                print(f"        {conflict['observation_2']['tool_name']}@{conflict['observation_2']['step']}: {val2}...")
    
    # 打印总体结果
    print("\n" + "="*60)
    print("总体统计结果")
    print("="*60)
    
    overall = result['overall']
    
    print("\n同工具一致率 (SameToolConsistency):")
    print(f"  总比较对数: {overall['same_tool']['total_pairs']}")
    print(f"  冲突对数: {overall['same_tool']['conflict_pairs']}")
    print(f"  一致率: {overall['same_tool']['consistency_rate']:.4f} ({overall['same_tool']['consistency_rate']*100:.2f}%)")
    
    print("\n跨工具一致率 (CrossToolConsistency):")
    print(f"  总比较对数: {overall['cross_tool']['total_pairs']}")
    print(f"  冲突对数: {overall['cross_tool']['conflict_pairs']}")
    print(f"  一致率: {overall['cross_tool']['consistency_rate']:.4f} ({overall['cross_tool']['consistency_rate']*100:.2f}%)")
    
    print("="*60)


if __name__ == "__main__":
    # 配置路径
    data_folders = [
        "/mnt/data/kw/ly/precision_index/data1",
        "/mnt/data/kw/ly/precision_index/data2",
    ]
    
    # 排除最后 N 个 steps（设为 2 排除最后两个总结 step，设为 0 不排除）
    exclude_last_steps = 0
    
    # 计算状态一致率
    result = calculate_state_consistency(data_folders, exclude_last_steps)
    
    # 打印结果
    print_results(result)
    
    # 保存结果到JSON文件
    output_file = "state_consistency_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_file}")
