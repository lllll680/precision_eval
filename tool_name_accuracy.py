import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple


def load_tool_schema(tool_schema_path: str) -> Set[str]:
    """
    从tool.txt文件中加载所有工具名称
    
    Args:
        tool_schema_path: tool.txt文件的路径
        
    Returns:
        包含所有工具名称的集合
    """
    tool_names = set()
    
    with open(tool_schema_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式提取所有工具名称
    # 匹配格式: "Name: tool_name"
    pattern = r'Name:\s*(\w+)'
    matches = re.findall(pattern, content)
    
    for match in matches:
        tool_names.add(match)
    
    return tool_names


def extract_tool_calls_from_response(response: List[Dict]) -> List[str]:
    """
    从response中提取所有的工具调用名称
    
    Args:
        response: JSON数据中的response字段
        
    Returns:
        所有调用的工具名称列表
    """
    tool_calls = []
    
    for step_dict in response:
        # 每个step_dict的格式是 {"step1": {...}, "step2": {...}}
        for step_key, step_value in step_dict.items():
            # step_value包含cot和coa
            if 'coa' in step_value:
                coa_list = step_value['coa']
                for coa_item in coa_list:
                    if 'action' in coa_item and 'name' in coa_item['action']:
                        tool_name = coa_item['action']['name']
                        tool_calls.append(tool_name)
    
    return tool_calls


def calculate_tool_name_accuracy(data_folders: List[str], tool_schema_path: str) -> Dict:
    """
    计算工具名正确率
    
    Args:
        data_folders: 数据文件夹路径列表，每个文件夹代表一个场景
        tool_schema_path: tool.txt文件的路径
        
    Returns:
        包含统计结果的字典，包括每条数据的结果和总体结果
    """
    # 加载工具schema
    valid_tool_names = load_tool_schema(tool_schema_path)
    print(f"加载了 {len(valid_tool_names)} 个有效工具")
    print(f"工具列表: {sorted(valid_tool_names)}\n")
    
    # 统计变量
    total_calls = 0  # 总调用数
    valid_calls = 0  # 合法调用数
    per_file_results = []  # 每条数据的结果
    
    # 遍历所有数据文件夹
    for data_folder in data_folders:
        data_path = Path(data_folder)
        
        if not data_path.exists():
            print(f"警告: 文件夹不存在，跳过: {data_folder}")
            continue
        
        print(f"处理数据文件夹: {data_folder}")
        
        # 遍历文件夹中的所有JSON文件
        json_files = list(data_path.glob('*.json'))
        
        for json_file in json_files:
            # 跳过特殊文件
            if json_file.name in ['question_info.json', 'batch_summary.json']:
                continue
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 针对每个文件的统计
                file_total_calls = 0
                file_valid_calls = 0
                file_invalid_details = []
                
                # 提取response中的所有工具调用
                if 'response' in data:
                    tool_calls = extract_tool_calls_from_response(data['response'])
                    
                    # 检查每个工具调用是否合法
                    for tool_name in tool_calls:
                        file_total_calls += 1
                        total_calls += 1
                        
                        if tool_name in valid_tool_names:
                            file_valid_calls += 1
                            valid_calls += 1
                        else:
                            file_invalid_details.append(tool_name)
                
                # 计算该文件的准确率
                file_accuracy = file_valid_calls / file_total_calls if file_total_calls > 0 else 0.0
                
                # 记录该文件的结果
                per_file_results.append({
                    'file': str(json_file),
                    'Acc_tool': file_accuracy,
                    'total_calls': file_total_calls,
                    'valid_calls': file_valid_calls,
                    'invalid_calls': file_total_calls - file_valid_calls,
                    'invalid_calls_details': file_invalid_details
                })
            
            except Exception as e:
                print(f"处理文件 {json_file} 时出错: {e}")
                continue
    
    # 计算总体准确率
    overall_accuracy = valid_calls / total_calls if total_calls > 0 else 0.0
    
    # 返回结果
    result = {
        'per_file_results': per_file_results,
        'overall': {
            'Acc_tool': overall_accuracy,
            'total_calls': total_calls,
            'valid_calls': valid_calls,
            'invalid_calls': total_calls - valid_calls
        }
    }
    
    return result


def print_results(result: Dict):
    """
    打印统计结果
    
    Args:
        result: calculate_tool_name_accuracy返回的结果字典
    """
    print("\n" + "="*60)
    print("工具名正确率统计结果")
    print("="*60)
    
    # 打印每条数据的结果
    print("\n每条数据的结果:")
    print("-"*60)
    for file_result in result['per_file_results']:
        print(f"\n文件: {file_result['file']}")
        print(f"  总调用数: {file_result['total_calls']}")
        print(f"  合法调用数: {file_result['valid_calls']}")
        print(f"  非法调用数: {file_result['invalid_calls']}")
        print(f"  Acc_tool: {file_result['Acc_tool']:.4f} ({file_result['Acc_tool']*100:.2f}%)")
        if file_result['invalid_calls_details']:
            print(f"  非法工具名: {', '.join(file_result['invalid_calls_details'])}")
    
    # 打印总体结果
    print("\n" + "="*60)
    print("总体统计结果")
    print("="*60)
    overall = result['overall']
    print(f"总调用数 (N_call): {overall['total_calls']}")
    print(f"合法调用数 (N_valid_tool): {overall['valid_calls']}")
    print(f"非法调用数: {overall['invalid_calls']}")
    print(f"工具名正确率 (Acc_tool): {overall['Acc_tool']:.4f} ({overall['Acc_tool']*100:.2f}%)")
    print("="*60)


if __name__ == "__main__":
    # 配置路径
    data_folders = [
        "/mnt/data/kw/ly/precision_index/data1",
        "/mnt/data/kw/ly/precision_index/data2",
    ]
    tool_schema_path = "/mnt/data/kw/ly/precision_index/tool.txt"
    
    # 计算工具名正确率
    result = calculate_tool_name_accuracy(data_folders, tool_schema_path)
    
    # 打印结果
    print_results(result)
    
    # 可选：保存结果到JSON文件
    output_file = " tool_name_accuracy_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_file}")
