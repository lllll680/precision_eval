import json
import os
import re
import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from jsonschema import validate, ValidationError, Draft7Validator


def extract_balanced_braces(text: str, start_pos: int = 0) -> Optional[str]:
    """
    提取从 start_pos 开始的平衡花括号内容
    
    Args:
        text: 要搜索的文本
        start_pos: 开始位置
        
    Returns:
        平衡的花括号内容（包含外层花括号），如果没有找到则返回 None
    """
    # 找到第一个 '{'
    brace_start = text.find('{', start_pos)
    if brace_start == -1:
        return None
    
    count = 0
    in_string = False
    escape_next = False
    
    for i in range(brace_start, len(text)):
        c = text[i]
        
        # 处理转义字符
        if escape_next:
            escape_next = False
            continue
        
        if c == '\\':
            escape_next = True
            continue
        
        # 处理字符串边界（简化处理，只考虑双引号和单引号）
        if c in ('"', "'") and not in_string:
            in_string = c
        elif c == in_string:
            in_string = False
        elif not in_string:
            if c == '{':
                count += 1
            elif c == '}':
                count -= 1
                if count == 0:
                    return text[brace_start:i+1]
    
    return None


def complete_output_schema(raw_schema: Dict) -> Dict:
    """
    补全不完整的Output schema为标准JSON Schema格式
    
    Args:
        raw_schema: 原始的不完整schema，格式如 {"field_name": {"type": "string"}}
        
    Returns:
        补全后的标准JSON Schema
    """
    if raw_schema is None:
        return None
    
    # 如果已经是标准格式（包含type字段），直接返回
    if "type" in raw_schema:
        return raw_schema
    
    # 否则，将其视为properties定义，补全为标准格式
    # 原始格式: {"field1": {"type": "string"}, "field2": {"type": "int"}}
    # 补全为标准格式:
    # {
    #     "type": "object",
    #     "properties": {...},
    #     "required": ["field1", "field2"],
    #     "additionalProperties": false
    # }
    
    completed_schema = {
        "type": "object",
        "properties": raw_schema,
        "required": list(raw_schema.keys()),  # 所有定义的字段都是必需的
        "additionalProperties": False  # 不允许额外字段
    }
    
    return completed_schema


def parse_tool_schema(tool_schema_path: str) -> Dict[str, Dict]:
    """
    从tool.txt文件中解析所有工具的schema信息
    
    Args:
        tool_schema_path: tool.txt文件的路径
        
    Returns:
        字典，key为工具名，value为包含input_schema和output_schema的字典
    """
    tools_schema = {}
    
    with open(tool_schema_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按工具分割（每个工具以数字+点开头）
    tool_blocks = re.split(r'\n(?=\d+\.\s+Name:)', content.strip())
    
    for block in tool_blocks:
        if not block.strip():
            continue
        
        # 提取工具名
        name_match = re.search(r'Name:\s*(\w+)', block)
        if not name_match:
            continue
        tool_name = name_match.group(1)
        
        # 提取Parameters (输入schema) - 使用平衡括号匹配
        input_schema = None
        params_pos = block.find('Parameters:')
        if params_pos != -1:
            params_str = extract_balanced_braces(block, params_pos)
            if params_str:
                try:
                    # 处理None值并解析
                    params_str_clean = params_str.replace('None', 'null')
                    input_schema = json.loads(params_str_clean.replace("'", '"'))
                except Exception as e:
                    print(f"警告: 解析工具 {tool_name} 的Parameters失败 (JSON): {e}")
                    try:
                        input_schema = ast.literal_eval(params_str)
                    except Exception as e2:
                        print(f"警告: 解析工具 {tool_name} 的Parameters失败 (AST): {e2}")
        
        # 提取Output (输出schema) - 使用平衡括号匹配
        output_schema = None
        output_pos = block.find('Output:')
        if output_pos != -1:
            output_str = extract_balanced_braces(block, output_pos)
            if output_str:
                try:
                    output_str_clean = output_str.replace('None', 'null')
                    output_schema = json.loads(output_str_clean.replace("'", '"'))
                except Exception as e:
                    print(f"警告: 解析工具 {tool_name} 的Output失败 (JSON): {e}")
                    try:
                        output_schema = ast.literal_eval(output_str)
                    except Exception as e2:
                        print(f"警告: 解析工具 {tool_name} 的Output失败 (AST): {e2}")
            else:
                print(f"警告: 工具 {tool_name} 的Output未找到有效的花括号内容")
        
        # 补全output_schema为标准JSON Schema格式
        output_schema = complete_output_schema(output_schema)
        
        tools_schema[tool_name] = {
            'input_schema': input_schema,
            'output_schema': output_schema
        }
    
    return tools_schema


def validate_against_schema(data: Any, schema: Dict) -> Tuple[bool, Optional[str]]:
    """
    使用JSON Schema验证数据
    
    Args:
        data: 要验证的数据
        schema: JSON Schema
        
    Returns:
        (是否合法, 错误信息)
    """
    if schema is None:
        return False, "Schema未定义或解析失败"
    
    try:
        # 创建验证器
        validator = Draft7Validator(schema)
        
        # 验证数据
        errors = list(validator.iter_errors(data))
        
        if errors:
            # 收集所有错误信息
            error_messages = [f"{e.message} at {'.'.join(str(p) for p in e.path)}" if e.path else e.message 
                            for e in errors]
            return False, "; ".join(error_messages)
        
        return True, None
    
    except Exception as e:
        return False, str(e)


def extract_tool_calls_with_details(response: List[Dict]) -> List[Dict]:
    """
    从response中提取所有的工具调用详情（包括args和observation）
    
    Args:
        response: JSON数据中的response字段
        
    Returns:
        包含工具调用详情的列表
    """
    tool_calls = []
    
    for step_dict in response:
        for step_key, step_value in step_dict.items():
            if 'coa' in step_value:
                coa_list = step_value['coa']
                for coa_item in coa_list:
                    if 'action' in coa_item:
                        action = coa_item['action']
                        observation = coa_item.get('observation', {})
                        
                        tool_calls.append({
                            'tool_name': action.get('name', ''),
                            'args': action.get('args', {}),
                            'observation': observation
                        })
    
    return tool_calls


def calculate_schema_validation_accuracy(data_folders: List[str], tool_schema_path: str) -> Dict:
    """
    计算输入与输出Schema合法率
    
    Args:
        data_folders: 数据文件夹路径列表
        tool_schema_path: tool.txt文件的路径
        
    Returns:
        包含统计结果的字典，包括每条数据的结果和总体结果
    """
    # 加载工具schema
    tools_schema = parse_tool_schema(tool_schema_path)
    print(f"加载了 {len(tools_schema)} 个工具的schema")
    print(f"工具列表: {sorted(tools_schema.keys())}\n")
    
    # 统计变量
    total_calls = 0  # 总调用数
    valid_calls = 0  # schema合法的调用数（action和observation都合法）
    action_valid_calls = 0  # action参数合法的调用数
    observation_valid_calls = 0  # observation合法的调用数
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
                file_action_valid_calls = 0
                file_observation_valid_calls = 0
                file_invalid_details = []
                
                # 提取response中的所有工具调用
                if 'response' in data:
                    tool_calls = extract_tool_calls_with_details(data['response'])
                    
                    # 检查每个工具调用的schema合法性
                    for call in tool_calls:
                        tool_name = call['tool_name']
                        args = call['args']
                        observation = call['observation']
                        
                        file_total_calls += 1
                        total_calls += 1
                        
                        # 获取该工具的schema
                        if tool_name not in tools_schema:
                            # 工具不存在，schema验证失败
                            file_invalid_details.append({
                                'tool_name': tool_name,
                                'reason': '工具不存在于schema中',
                                'action_valid': False,
                                'observation_valid': False
                            })
                            continue
                        
                        tool_schema = tools_schema[tool_name]
                        input_schema = tool_schema['input_schema']
                        output_schema = tool_schema['output_schema']
                        
                        # 验证args
                        args_valid, args_error = validate_against_schema(args, input_schema)
                        
                        # 验证observation
                        obs_valid, obs_error = validate_against_schema(observation, output_schema)
                        
                        # 构建调用详情（无论是否合法都记录）
                        call_info = {
                            'tool_name': tool_name,
                            'action_valid': args_valid,
                            'observation_valid': obs_valid
                        }
                        if not args_valid:
                            call_info['action_error'] = args_error
                        if not obs_valid:
                            call_info['observation_error'] = obs_error
                        file_invalid_details.append(call_info)
                        
                        # 分别统计 action 和 observation 的合法数
                        if args_valid:
                            file_action_valid_calls += 1
                            action_valid_calls += 1
                        if obs_valid:
                            file_observation_valid_calls += 1
                            observation_valid_calls += 1
                        
                        # 只有args和observation都合法才算合法
                        if args_valid and obs_valid:
                            file_valid_calls += 1
                            valid_calls += 1
                
                # 计算该文件的准确率
                file_accuracy = file_valid_calls / file_total_calls if file_total_calls > 0 else 0.0
                
                # 记录该文件的结果
                per_file_results.append({
                    'file': str(json_file),
                    'Acc_schema': file_accuracy,
                    'total_calls': file_total_calls,
                    'valid_calls': file_valid_calls,
                    'invalid_calls': file_total_calls - file_valid_calls,
                    'action_valid_calls': file_action_valid_calls,
                    'observation_valid_calls': file_observation_valid_calls,
                    'calls_details': file_invalid_details
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
            'Acc_schema': overall_accuracy,
            'total_calls': total_calls,
            'valid_calls': valid_calls,
            'invalid_calls': total_calls - valid_calls,
            'action_valid_calls': action_valid_calls,
            'observation_valid_calls': observation_valid_calls
        }
    }
    
    return result


def print_results(result: Dict):
    """
    打印统计结果
    
    Args:
        result: calculate_schema_validation_accuracy返回的结果字典
    """
    print("\n" + "="*60)
    print("输入与输出Schema合法率统计结果")
    print("="*60)
    
    # 打印每条数据的结果
    print("\n每条数据的结果:")
    print("-"*60)
    for file_result in result['per_file_results']:
        print(f"\n文件: {file_result['file']}")
        print(f"  总调用数: {file_result['total_calls']}")
        print(f"  合法调用数: {file_result['valid_calls']}")
        print(f"  非法调用数: {file_result['invalid_calls']}")
        print(f"  Action合法数: {file_result['action_valid_calls']}")
        print(f"  Observation合法数: {file_result['observation_valid_calls']}")
        print(f"  Acc_schema: {file_result['Acc_schema']:.4f} ({file_result['Acc_schema']*100:.2f}%)")
        
        if file_result['calls_details']:
            print(f"  调用详情:")
            for detail in file_result['calls_details']:
                action_status = "✓" if detail.get('action_valid', False) else "✗"
                obs_status = "✓" if detail.get('observation_valid', False) else "✗"
                print(f"    - 工具: {detail['tool_name']} [Action:{action_status}, Obs:{obs_status}]")
                if 'reason' in detail:
                    print(f"      原因: {detail['reason']}")
                else:
                    if not detail['action_valid']:
                        print(f"      Action错误: {detail.get('action_error', 'Unknown error')}")
                    if not detail['observation_valid']:
                        print(f"      Observation错误: {detail.get('observation_error', 'Unknown error')}")
    
    # 打印总体结果
    print("\n" + "="*60)
    print("总体统计结果")
    print("="*60)
    overall = result['overall']
    print(f"总调用数 (N_call): {overall['total_calls']}")
    print(f"合法调用数 (N_schema_valid): {overall['valid_calls']}")
    print(f"非法调用数: {overall['invalid_calls']}")
    print(f"Action合法数: {overall['action_valid_calls']}")
    print(f"Observation合法数: {overall['observation_valid_calls']}")
    print(f"Schema合法率 (Acc_schema): {overall['Acc_schema']:.4f} ({overall['Acc_schema']*100:.2f}%)")
    print("="*60)


if __name__ == "__main__":
    # 配置路径
    data_folders = [
        "/mnt/data/kw/ly/precision_index/data1",
        "/mnt/data/kw/ly/precision_index/data2",
    ]
    tool_schema_path = "/mnt/data/kw/ly/precision_index/tool.txt"
    
    # 计算Schema合法率
    result = calculate_schema_validation_accuracy(data_folders, tool_schema_path)
    
    # 打印结果
    print_results(result)
    
    # 可选：保存结果到JSON文件
    output_file = "schema_validation_accuracy_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_file}")
