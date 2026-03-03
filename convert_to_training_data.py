#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据转换脚本 - 将JSON数据转换为对话格式的训练数据

功能：
- 遍历多个文件夹中的JSON文件
- 将每个JSON文件转换为对话格式
- 第一个user消息：query + 工具描述
- 每个step的cot+action作为assistant回复
- 每个step的observation作为下一个user消息

输出格式：
[
    {"role": "user", "content": "<query + tools>"},
    {"role": "assistant", "content": "<step1 cot + action>"},
    {"role": "user", "content": "<step1 observation>"},
    {"role": "assistant", "content": "<step2 cot + action>"},
    ...
]

使用方法：
    python convert_to_training_data.py
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any


def load_tool_descriptions(tool_file_path: str) -> str:
    """
    加载工具描述文件
    
    Args:
        tool_file_path: 工具描述文件路径
        
    Returns:
        工具描述文本
    """
    try:
        with open(tool_file_path, 'r', encoding='utf-8') as f:
            tool_text = f.read()
        return tool_text
    except Exception as e:
        print(f"❌ 加载工具描述文件失败: {e}")
        return ""


def format_action(action: Dict) -> str:
    """
    格式化action为字符串
    
    Args:
        action: action字典，包含name和args
        
    Returns:
        格式化后的action字符串
    """
    name = action.get('name', '')
    args = action.get('args', {})
    
    # 格式化为JSON字符串
    action_str = json.dumps({
        "name": name,
        "args": args
    }, ensure_ascii=False, indent=2)
    
    return action_str


def format_observation(observation: Dict) -> str:
    """
    格式化observation为字符串
    
    Args:
        observation: observation字典
        
    Returns:
        格式化后的observation字符串
    """
    return json.dumps(observation, ensure_ascii=False, indent=2)


def convert_json_to_conversation(json_data: Dict, tool_descriptions: str) -> List[Dict]:
    """
    将单个JSON文件转换为对话格式
    
    Args:
        json_data: JSON数据
        tool_descriptions: 工具描述文本
        
    Returns:
        对话格式的消息列表
    """
    messages = []
    
    # 获取query
    query = json_data.get('query', '')
    
    # 第一个user消息：query + 工具描述
    first_user_content = f"{query}\n\n可用工具：\n{tool_descriptions}"
    messages.append({
        "role": "user",
        "content": first_user_content
    })
    
    # 处理response中的每个step
    response = json_data.get('response', [])
    
    for step_item in response:
        # step_item格式: {"step1": {"cot": "...", "coa": [...]}}
        for step_key, step_data in step_item.items():
            if not step_key.startswith('step'):
                continue
            
            cot = step_data.get('cot', '')
            coa_list = step_data.get('coa', [])
            
            # 处理每个coa（可能有多个action-observation对）
            for coa_item in coa_list:
                action = coa_item.get('action', {})
                observation = coa_item.get('observation', {})
                
                # assistant消息：cot + action
                action_str = format_action(action)
                assistant_content = f"Think：{cot}\n\n action:\n{action_str}"
                messages.append({
                    "role": "assistant",
                    "content": assistant_content
                })
                
                # user消息：observation
                observation_str = format_observation(observation)
                user_content = f"observation: \n{observation_str}"
                messages.append({
                    "role": "user",
                    "content": user_content
                })
    
    return messages


def process_data_folders(
    data_folders: List[str],
    tool_file_path: str,
    output_file: str = 'training_data.jsonl'
):
    """
    处理多个数据文件夹，生成训练数据
    
    Args:
        data_folders: 数据文件夹路径列表
        tool_file_path: 工具描述文件路径
        output_file: 输出文件路径
    """
    # 加载工具描述
    print("正在加载工具描述...")
    tool_descriptions = load_tool_descriptions(tool_file_path)
    if not tool_descriptions:
        print("⚠️  工具描述为空，将继续处理但不包含工具信息")
    
    print(f"工具描述加载完成，共 {len(tool_descriptions)} 个字符\n")
    
    # 统计信息
    total_files = 0
    total_conversations = 0
    
    # 打开输出文件
    with open(output_file, 'w', encoding='utf-8') as out_f:
        # 遍历每个文件夹
        for folder_path in data_folders:
            folder = Path(folder_path)
            
            if not folder.exists():
                print(f"⚠️  文件夹不存在，跳过: {folder_path}")
                continue
            
            print(f"处理文件夹: {folder_path}")
            
            # 获取所有JSON文件
            json_files = []
            for json_file in sorted(folder.glob("*.json")):
                # 跳过特殊文件
                if json_file.name in ['question_info.json', 'batch_summary.json']:
                    continue
                json_files.append(json_file)
            
            print(f"  找到 {len(json_files)} 个JSON文件")
            
            # 处理每个JSON文件
            for json_file in json_files:
                try:
                    # 读取JSON文件
                    with open(json_file, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    # 转换为对话格式
                    messages = convert_json_to_conversation(json_data, tool_descriptions)
                    
                    # 构建训练样本
                    training_sample = {
                        "messages": messages,
                        "source_file": str(json_file)
                    }
                    
                    # 写入输出文件（JSONL格式，每行一个JSON对象）
                    out_f.write(json.dumps(training_sample, ensure_ascii=False) + '\n')
                    
                    total_files += 1
                    total_conversations += len(messages)
                    
                    print(f"  ✅ {json_file.name}: {len(messages)} 条消息")
                    
                except Exception as e:
                    print(f"  ❌ 处理文件 {json_file.name} 时出错: {e}")
                    continue
            
            print()
    
    # 打印统计信息
    print("="*60)
    print("数据转换完成")
    print("="*60)
    print(f"处理文件数: {total_files}")
    print(f"生成消息数: {total_conversations}")
    print(f"输出文件: {output_file}")
    print("="*60)


def preview_sample(output_file: str, num_samples: int = 1):
    """
    预览生成的训练数据样本
    
    Args:
        output_file: 输出文件路径
        num_samples: 预览样本数量
    """
    print("\n" + "="*60)
    print("预览训练数据样本")
    print("="*60)
    
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                
                sample = json.loads(line)
                print(f"\n样本 {i+1}:")
                print(f"来源文件: {sample.get('source_file', 'unknown')}")
                print(f"消息数量: {len(sample.get('messages', []))}")
                print("\n消息内容:")
                
                for j, msg in enumerate(sample.get('messages', [])[:6]):  # 只显示前6条
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    # 截断过长的内容
                    if len(content) > 200:
                        content = content[:200] + "..."
                    print(f"\n  [{j+1}] {role}:")
                    print(f"  {content}")
                
                if len(sample.get('messages', [])) > 6:
                    print(f"\n  ... 还有 {len(sample.get('messages', [])) - 6} 条消息")
                
                print("\n" + "-"*60)
    
    except Exception as e:
        print(f"❌ 预览失败: {e}")


if __name__ == "__main__":
    # 配置参数
    data_folders = [
        "/Users/liaoying/Desktop/研一/llm/data_eval/precision_index/data1",
        "/Users/liaoying/Desktop/研一/llm/data_eval/precision_index/data2",
        # 添加更多文件夹...
    ]
    
    tool_file_path = "/Users/liaoying/Desktop/研一/llm/data_eval/precision_index/tool.txt"
    output_file = "/Users/liaoying/Desktop/研一/llm/data_eval/precision_index/training_data.jsonl"
    
    # 执行转换
    process_data_folders(data_folders, tool_file_path, output_file)
    
    # 预览样本
    preview_sample(output_file, num_samples=1)
