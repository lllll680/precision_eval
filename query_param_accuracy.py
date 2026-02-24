import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_entities_from_query(query: str, model, tokenizer) -> Dict[str, List[str]]:
    """
    使用Qwen模型从query中提取结构化实体
    
    Args:
        query: 用户查询文本
        model: Qwen模型
        tokenizer: Qwen tokenizer
        
    Returns:
        提取的实体字典，格式如 {"device_name": ["device1"], "interface_name": ["eth0"], "ip_address": ["192.168.1.1"]}
    """
    # 构建提示词
    prompt = f"""请从以下网络运维查询中提取结构化实体信息。只提取明确出现的实体，不要推测。

查询：{query}

请以JSON格式输出提取的实体，包括以下类型（如果存在）：
- device_name: 设备名称（如 serverleaf01_1_16.135）
- interface_name: 接口名称（如 10GE1/0/24）
- ip_address: IP地址（如 192.168.100.2）
- bgp_neighbor: BGP邻居地址
- 其他相关实体

输出格式示例：
{{"device_name": ["serverleaf01_1_16.135"], "interface_name": ["10GE1/0/24"]}}

只输出JSON，不要其他解释："""

    # 生成回复
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        do_sample=True
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # 解析JSON响应
    try:
        # 尝试提取JSON部分
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            entities = json.loads(json_match.group())
            # 确保所有值都是列表
            for key in entities:
                if not isinstance(entities[key], list):
                    entities[key] = [entities[key]]
            return entities
        else:
            print(f"警告: 无法从模型响应中提取JSON: {response}")
            return {}
    except Exception as e:
        print(f"警告: 解析实体JSON失败: {e}, 响应: {response}")
        return {}


def check_param_match(args: Dict, entities: Dict[str, List[str]]) -> Tuple[int, int, List[Dict]]:
    """
    检查args中的参数值是否与query中提取的实体匹配
    
    核心逻辑：
    - 只有当query中存在某类型实体时，才检查对应参数是否正确引用
    - 如果query中没有该类型实体，则该参数不计入统计（可能来自中间结果）
    - 检测跨类型误用（如把device_name当作ip_address使用）
    
    Args:
        args: action中的参数
        entities: 从query中提取的实体
        
    Returns:
        (正确匹配数, 需要检查的参数总数, 不匹配的详情列表)
    """
    correct_count = 0
    total_count = 0
    mismatch_details = []
    
    # 定义参数名到期望实体类型的映射
    param_to_entity_mapping = {
        'device_name': 'device_name',
        'device_id': 'device_name',
        'host_name': 'device_name',
        'interface_name': 'interface_name',
        'interface_id': 'interface_name',
        'ip_address': 'ip_address',
        'source_ip': 'ip_address',
        'dest_ip': 'ip_address',
        'bgp_neighbor': 'bgp_neighbor',
        'neighbor_ip': 'bgp_neighbor',
    }
    
    # 收集所有query中的实体值，用于检测跨类型误用
    all_entity_values = {}  # value -> entity_type
    for entity_type, values in entities.items():
        for v in values:
            all_entity_values[v] = entity_type
    
    for param_name, param_value in args.items():
        # 检查该参数是否需要与query实体匹配
        if param_name not in param_to_entity_mapping:
            continue
        
        expected_entity_type = param_to_entity_mapping[param_name]
        
        # 关键：只有当query中存在该类型实体时才计入统计
        if expected_entity_type not in entities or not entities[expected_entity_type]:
            # query中没有该类型实体，检查是否存在跨类型误用
            if param_value in all_entity_values:
                actual_type = all_entity_values[param_value]
                # 跨类型误用：参数期望ip_address，但值来自device_name
                mismatch_details.append({
                    'param_name': param_name,
                    'param_value': param_value,
                    'expected_type': expected_entity_type,
                    'expected_values': [],
                    'error_type': 'cross_type_misuse',
                    'actual_type': actual_type,
                    'reason': f'跨类型误用：参数期望{expected_entity_type}，但值来自query中的{actual_type}'
                })
                total_count += 1  # 跨类型误用也计入统计
            # 否则：query中无该类型实体，且值也不来自query，不计入统计
            continue
        
        # query中有该类型实体，计入统计
        total_count += 1
        
        # 检查参数值是否在期望的实体列表中
        if param_value in entities[expected_entity_type]:
            correct_count += 1
        else:
            # 检查是否是跨类型误用
            if param_value in all_entity_values:
                actual_type = all_entity_values[param_value]
                mismatch_details.append({
                    'param_name': param_name,
                    'param_value': param_value,
                    'expected_type': expected_entity_type,
                    'expected_values': entities[expected_entity_type],
                    'error_type': 'cross_type_misuse',
                    'actual_type': actual_type,
                    'reason': f'跨类型误用：参数期望{expected_entity_type}，但值来自query中的{actual_type}'
                })
            else:
                mismatch_details.append({
                    'param_name': param_name,
                    'param_value': param_value,
                    'expected_type': expected_entity_type,
                    'expected_values': entities[expected_entity_type],
                    'error_type': 'wrong_value',
                    'actual_type': None,
                    'reason': f'参数值与query中的{expected_entity_type}不一致'
                })
    
    return correct_count, total_count, mismatch_details


def load_schema_validation_results(schema_result_path: str) -> Dict[str, Dict[str, Any]]:
    """
    加载schema验证结果，构建文件->工具调用有效性的映射
    
    只基于action_valid判断（不涉及observation），用于联合指标计算
    
    Args:
        schema_result_path: schema验证结果JSON文件路径
        
    Returns:
        字典格式: {file_path: {'_action_invalid_tools': set(), ...}}
    """
    valid_calls_map = {}
    
    try:
        with open(schema_result_path, 'r', encoding='utf-8') as f:
            schema_result = json.load(f)
        
        for file_result in schema_result.get('per_file_results', []):
            file_path = file_result['file']
            valid_calls_map[file_path] = {}
            
            # 统计每个文件中哪些工具的action验证失败
            # 只基于action_valid判断，不涉及observation
            action_invalid_tools = set()
            for detail in file_result.get('invalid_calls_details', []):
                tool_name = detail.get('tool_name', '')
                action_valid = detail.get('action_valid', True)
                if not action_valid:
                    action_invalid_tools.add(tool_name)
            
            valid_calls_map[file_path]['_action_invalid_tools'] = action_invalid_tools
            
    except Exception as e:
        print(f"警告: 加载schema验证结果失败: {e}")
    
    return valid_calls_map


def calculate_query_param_accuracy(data_folders: List[str], model_path: str, 
                                    schema_result_path: Optional[str] = None) -> Dict:
    """
    计算Query参数引用正确率
    
    Args:
        data_folders: 数据文件夹路径列表
        model_path: Qwen模型路径
        schema_result_path: schema验证结果文件路径（可选），用于计算联合指标
        
    Returns:
        包含统计结果的字典，包括每条数据的结果和总体结果
    """
    # 加载schema验证结果（如果提供）
    schema_valid_map = {}
    if schema_result_path:
        print(f"加载schema验证结果: {schema_result_path}")
        schema_valid_map = load_schema_validation_results(schema_result_path)
    
    # 加载Qwen模型
    print(f"正在加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    print("模型加载完成\n")
    
    # 统计变量 - 独立指标（所有calls）
    total_params = 0  # 需要检查的参数总数
    correct_params = 0  # 正确引用的参数数
    cross_type_errors = 0  # 跨类型误用错误数
    wrong_value_errors = 0  # 值不一致错误数
    
    # 统计变量 - 联合指标（仅schema有效的calls）
    schema_valid_total_params = 0
    schema_valid_correct_params = 0
    schema_valid_cross_type_errors = 0
    schema_valid_wrong_value_errors = 0
    
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
                file_total_params = 0
                file_correct_params = 0
                file_mismatch_details = []
                
                # 联合指标 - 仅action有效的calls
                file_schema_valid_total = 0
                file_schema_valid_correct = 0
                
                # 获取该文件的schema验证信息
                file_path_str = str(json_file)
                file_schema_info = schema_valid_map.get(file_path_str, {})
                
                # 提取query中的实体
                query = data.get('query', '')
                if not query:
                    print(f"警告: 文件 {json_file} 中没有query字段")
                    continue
                
                print(f"  处理文件: {json_file.name}")
                entities = extract_entities_from_query(query, model, tokenizer)
                print(f"    提取的实体: {entities}")
                
                # 检查response中的所有action参数
                if 'response' in data:
                    for step_dict in data['response']:
                        for step_key, step_value in step_dict.items():
                            if 'coa' in step_value:
                                coa_list = step_value['coa']
                                for coa_item in coa_list:
                                    if 'action' in coa_item:
                                        action = coa_item['action']
                                        tool_name = action.get('name', '')
                                        args = action.get('args', {})
                                        
                                        # 检查参数匹配
                                        correct, total, mismatches = check_param_match(args, entities)
                                        
                                        # 独立指标统计（所有calls）
                                        file_total_params += total
                                        file_correct_params += correct
                                        total_params += total
                                        correct_params += correct
                                        
                                        # 判断该call的action是否schema有效（不涉及observation）
                                        action_invalid_tools = file_schema_info.get('_action_invalid_tools', set())
                                        is_action_valid = tool_name not in action_invalid_tools
                                        
                                        # 联合指标统计（仅action有效的calls，不涉及observation）
                                        if schema_result_path and is_action_valid:
                                            file_schema_valid_total += total
                                            file_schema_valid_correct += correct
                                            schema_valid_total_params += total
                                            schema_valid_correct_params += correct
                                        
                                        if mismatches:
                                            for mismatch in mismatches:
                                                mismatch['tool_name'] = tool_name
                                                mismatch['step'] = step_key
                                                mismatch['action_valid'] = is_action_valid
                                                # 统计错误类型
                                                if mismatch.get('error_type') == 'cross_type_misuse':
                                                    cross_type_errors += 1
                                                    if schema_result_path and is_action_valid:
                                                        schema_valid_cross_type_errors += 1
                                                elif mismatch.get('error_type') == 'wrong_value':
                                                    wrong_value_errors += 1
                                                    if schema_result_path and is_action_valid:
                                                        schema_valid_wrong_value_errors += 1
                                            file_mismatch_details.extend(mismatches)
                
                # 计算该文件的准确率
                file_accuracy = file_correct_params / file_total_params if file_total_params > 0 else 1.0
                file_schema_valid_accuracy = file_schema_valid_correct / file_schema_valid_total if file_schema_valid_total > 0 else 0.0
                
                # 记录该文件的结果
                file_result = {
                    'file': str(json_file),
                    'query': query,
                    'extracted_entities': entities,
                    'Acc_param_query': file_accuracy,
                    'total_params': file_total_params,
                    'correct_params': file_correct_params,
                    'incorrect_params': file_total_params - file_correct_params,
                    'mismatch_details': file_mismatch_details
                }
                
                # 如果启用联合指标，添加相关统计
                if schema_result_path:
                    file_result['schema_valid_metrics'] = {
                        'Acc_param_query_schema_valid': file_schema_valid_accuracy,
                        'total_params': file_schema_valid_total,
                        'correct_params': file_schema_valid_correct,
                        'incorrect_params': file_schema_valid_total - file_schema_valid_correct
                    }
                
                per_file_results.append(file_result)
            
            except Exception as e:
                print(f"处理文件 {json_file} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 计算总体准确率
    overall_accuracy = correct_params / total_params if total_params > 0 else 1.0
    schema_valid_accuracy = schema_valid_correct_params / schema_valid_total_params if schema_valid_total_params > 0 else 0.0
    
    # 返回结果
    result = {
        'per_file_results': per_file_results,
        'overall': {
            'Acc_param_query': overall_accuracy,
            'total_params': total_params,
            'correct_params': correct_params,
            'incorrect_params': total_params - correct_params,
            'error_breakdown': {
                'cross_type_misuse': cross_type_errors,
                'wrong_value': wrong_value_errors
            }
        }
    }
    
    # 如果启用联合指标，添加相关统计
    if schema_result_path:
        result['overall_schema_valid'] = {
            'Acc_param_query_schema_valid': schema_valid_accuracy,
            'total_params': schema_valid_total_params,
            'correct_params': schema_valid_correct_params,
            'incorrect_params': schema_valid_total_params - schema_valid_correct_params,
            'error_breakdown': {
                'cross_type_misuse': schema_valid_cross_type_errors,
                'wrong_value': schema_valid_wrong_value_errors
            }
        }
    
    return result


def print_results(result: Dict):
    """
    打印统计结果
    
    Args:
        result: calculate_query_param_accuracy返回的结果字典
    """
    print("\n" + "="*60)
    print("Query参数引用正确率统计结果")
    print("="*60)
    
    # 打印每条数据的结果
    print("\n每条数据的结果:")
    print("-"*60)
    for file_result in result['per_file_results']:
        print(f"\n文件: {file_result['file']}")
        print(f"  Query: {file_result['query']}")
        print(f"  提取的实体: {file_result['extracted_entities']}")
        print(f"  需检查参数数: {file_result['total_params']}")
        print(f"  正确参数数: {file_result['correct_params']}")
        print(f"  错误参数数: {file_result['incorrect_params']}")
        print(f"  Acc_param_query: {file_result['Acc_param_query']:.4f} ({file_result['Acc_param_query']*100:.2f}%)")
        
        # 显示该文件的联合指标
        if 'schema_valid_metrics' in file_result:
            svm = file_result['schema_valid_metrics']
            print(f"  [联合指标-仅Action有效] 检查参数数: {svm['total_params']}, 正确: {svm['correct_params']}, "
                  f"Acc: {svm['Acc_param_query_schema_valid']:.4f} ({svm['Acc_param_query_schema_valid']*100:.2f}%)")
        
        if file_result['mismatch_details']:
            print(f"  参数不匹配详情:")
            for detail in file_result['mismatch_details']:
                error_type = detail.get('error_type', 'unknown')
                error_type_cn = {
                    'cross_type_misuse': '跨类型误用',
                    'wrong_value': '值不一致'
                }.get(error_type, error_type)
                
                print(f"    - Step: {detail['step']}, 工具: {detail['tool_name']}")
                print(f"      错误类型: {error_type_cn}")
                print(f"      参数名: {detail['param_name']}, 参数值: {detail['param_value']}")
                print(f"      期望类型: {detail.get('expected_type', 'N/A')}, 期望值: {detail['expected_values']}")
                if detail.get('actual_type'):
                    print(f"      实际来源类型: {detail['actual_type']}")
                print(f"      原因: {detail['reason']}")
    
    # 打印总体结果
    print("\n" + "="*60)
    print("总体统计结果")
    print("="*60)
    overall = result['overall']
    print(f"需检查参数总数 (N_query_param): {overall['total_params']}")
    print(f"正确引用参数数 (N_correct_query): {overall['correct_params']}")
    print(f"错误引用参数数: {overall['incorrect_params']}")
    
    # 打印错误类型分布
    if 'error_breakdown' in overall:
        breakdown = overall['error_breakdown']
        print(f"  - 跨类型误用 (cross_type_misuse): {breakdown.get('cross_type_misuse', 0)}")
        print(f"  - 值不一致 (wrong_value): {breakdown.get('wrong_value', 0)}")
    
    print(f"Query参数引用正确率 (Acc_param_query): {overall['Acc_param_query']:.4f} ({overall['Acc_param_query']*100:.2f}%)")
    
    # 打印联合指标结果（仅schema有效的calls）
    if 'overall_schema_valid' in result:
        print("\n" + "-"*60)
        print("联合指标 - 仅Schema有效的Calls")
        print("-"*60)
        schema_valid = result['overall_schema_valid']
        print(f"需检查参数总数 (Schema有效): {schema_valid['total_params']}")
        print(f"正确引用参数数: {schema_valid['correct_params']}")
        print(f"错误引用参数数: {schema_valid['incorrect_params']}")
        
        if 'error_breakdown' in schema_valid:
            breakdown = schema_valid['error_breakdown']
            print(f"  - 跨类型误用 (cross_type_misuse): {breakdown.get('cross_type_misuse', 0)}")
            print(f"  - 值不一致 (wrong_value): {breakdown.get('wrong_value', 0)}")
        
        print(f"Query参数引用正确率 (Schema有效): {schema_valid['Acc_param_query_schema_valid']:.4f} ({schema_valid['Acc_param_query_schema_valid']*100:.2f}%)")
    
    print("="*60)


if __name__ == "__main__":
    # 配置路径
    data_folders = [
        "/mnt/data/kw/ly/precision_index/data1",
        "/mnt/data/kw/ly/precision_index/data2",
        # 添加更多数据文件夹路径
    ]
    
    # Qwen模型路径（需要填写实际路径）
    model_path = "/mnt/data/kw/models/Qwen/Qwen2.5-7B-Instruct"  # TODO: 填写实际的Qwen3-8b模型路径
    
    # Schema验证结果路径（用于计算联合指标）
    schema_result_path = "/mnt/data/kw/ly/precision_index/schema_validation_accuracy_result.json"
    
    # 计算Query参数引用正确率（同时计算独立指标和联合指标）
    result = calculate_query_param_accuracy(data_folders, model_path, schema_result_path)
    
    # 打印结果
    print_results(result)
    
    # 保存结果到JSON文件
    output_file = "query_param_accuracy_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_file}")
