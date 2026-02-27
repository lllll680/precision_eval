import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_balanced_json(text: str, start_char: str = '{', end_char: str = '}') -> Optional[str]:
    """
    从文本中提取第一个平衡的JSON结构（对象或数组）
    
    Args:
        text: 包含JSON的文本
        start_char: 起始字符，'{' 或 '['
        end_char: 结束字符，'}' 或 ']'
        
    Returns:
        第一个完整的JSON字符串，如果没有找到则返回 None
    """
    start_pos = text.find(start_char)
    if start_pos == -1:
        return None
    
    count = 0
    in_string = False
    escape_next = False
    
    for i in range(start_pos, len(text)):
        c = text[i]
        
        if escape_next:
            escape_next = False
            continue
        
        if c == '\\':
            escape_next = True
            continue
        
        if c == '"' and not in_string:
            in_string = True
        elif c == '"' and in_string:
            in_string = False
        elif not in_string:
            if c == start_char:
                count += 1
            elif c == end_char:
                count -= 1
                if count == 0:
                    return text[start_pos:i+1]
    
    return None


def extract_json_robust(text: str) -> Optional[Any]:
    """
    健壮的JSON提取，支持多种情况：
    1. 直接输出JSON
    2. 包含<think>...</think>标签
    3. 包含未闭合的<think>标签
    4. JSON对象或数组
    
    Args:
        text: 模型输出的文本
        
    Returns:
        解析后的JSON对象，如果失败则返回 None
    """
    if not text or not text.strip():
        return None
    
    # 策略1: 直接尝试解析
    try:
        return json.loads(text.strip())
    except:
        pass
    
    # 策略2: 清理各种 think 标签模式
    cleaned = text
    
    # 模式1: <think>...</think> 或 <thinking>...</thinking> (有闭合标签)
    cleaned = re.sub(r'<think[^>]*>.*?</think[^>]*>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'<thinking[^>]*>.*?</thinking[^>]*>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    # 模式2: <think>... 无闭合标签（截取到第一个JSON开始位置）
    think_match = re.search(r'<think[^>]*>', cleaned, re.IGNORECASE)
    if think_match:
        # 找到JSON开始位置
        json_start = -1
        for char in ['{', '[']:
            pos = cleaned.find(char, think_match.end())
            if pos != -1 and (json_start == -1 or pos < json_start):
                json_start = pos
        
        if json_start != -1:
            cleaned = cleaned[json_start:]
        else:
            # 没有找到JSON，尝试删除think标签到末尾
            cleaned = cleaned[:think_match.start()]
    
    cleaned = cleaned.strip()
    
    # 策略3: 尝试提取JSON对象
    json_str = extract_balanced_json(cleaned, '{', '}')
    if json_str:
        try:
            return json.loads(json_str)
        except:
            pass
    
    # 策略4: 尝试提取JSON数组
    json_str = extract_balanced_json(cleaned, '[', ']')
    if json_str:
        try:
            return json.loads(json_str)
        except:
            pass
    
    # 策略5: 尝试从原文本直接提取（跳过think标签前的内容）
    json_str = extract_balanced_json(text, '{', '}')
    if json_str:
        try:
            return json.loads(json_str)
        except:
            pass
    
    json_str = extract_balanced_json(text, '[', ']')
    if json_str:
        try:
            return json.loads(json_str)
        except:
            pass
    
    return None


def load_entities_from_config(config_path: str) -> Dict[str, List[str]]:
    """
    从配置文件加载每个文件夹的实体列表
    
    Args:
        config_path: 配置文件路径，JSON格式
        
    Returns:
        {folder_name: [entity1, entity2, ...]}
        
    配置文件格式示例:
    {
        "data1": ["192.168.100.2", "borderleaf01", "10GE1/0/24"],
        "data2": ["10.84.21.109", "serverleaf01"]
    }
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"成功加载实体配置文件: {config_path}")
        return config
    except Exception as e:
        print(f"警告: 加载实体配置文件失败: {e}")
        return {}


def extract_entities_from_query(query: str, model, tokenizer) -> List[str]:
    """
    使用Qwen模型从query中提取所有实体值（不区分类型）
    
    Args:
        query: 用户查询文本
        model: Qwen模型
        tokenizer: Qwen tokenizer
        
    Returns:
        提取的实体值列表，如 ["192.168.100.2", "borderleaf01", "10GE1/0/24"]
    """
    # 构建prompt - 简化为提取所有实体值
    prompt = f"""从以下网络运维查询中提取所有具体的实体值，包括：
- IP地址（如 192.168.100.2, 10.84.21.109）
- 设备名称（如 borderleaf01_1_16.140, serverleaf01）
- 接口名称（如 10GE1/0/24, Eth-Trunk1）
- VLAN ID、端口号等其他具体数值

查询：{query}

要求：
1. 只提取明确出现的具体值，不要推测
2. 输出一个简单的列表，不区分类型
3. 只输出JSON，不要解释

输出格式：{{"entities": ["值1", "值2", ...]}}

示例：
查询："192.168.100.2和192.168.180.2之间网络不通"
输出：{{"entities": ["192.168.100.2", "192.168.180.2"]}}

查询："在borderleaf01_1_16.140设备上BGP邻居10.84.21.109建立失败"
输出：{{"entities": ["borderleaf01_1_16.140", "10.84.21.109"]}}

JSON输出："""

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
    
    # 使用健壮的JSON提取
    try:
        result = extract_json_robust(response)
        
        if result is None:
            print(f"警告: 无法从模型响应中提取JSON: {response[:200]}")
            return []
        
        # 支持两种格式:
        # 1. {"entities": ["val1", "val2"]} - 新格式
        # 2. ["val1", "val2"] - 纯数组格式
        # 3. {"type1": ["val1"], "type2": ["val2"]} - 旧格式（兼容）
        
        if isinstance(result, list):
            # 纯数组格式
            return [str(v) for v in result]
        elif isinstance(result, dict):
            if 'entities' in result:
                # 新格式
                entities = result['entities']
                if isinstance(entities, list):
                    return [str(v) for v in entities]
                else:
                    return [str(entities)]
            else:
                # 旧格式 - 合并所有值
                all_values = []
                for key, values in result.items():
                    if isinstance(values, list):
                        all_values.extend([str(v) for v in values])
                    else:
                        all_values.append(str(values))
                return all_values
        else:
            return [str(result)]
            
    except Exception as e:
        print(f"警告: 解析实体JSON失败: {e}, 响应: {response[:200]}")
        return []


# 需要检查的参数名列表（这些参数应该来自 Query 中的实体）
QUERY_RELATED_PARAMS = {
    'device_name', 'device_id', 'host_name', 'hostname',
    'interface_name', 'interface_id', 'port_name',
    'ip_address', 'ip', 'source_ip', 'dest_ip', 'target_ip', 'destination_ip',
    'bgp_neighbor', 'neighbor_ip', 'peer_ip',
    'vlan_id', 'vlan',
}


def check_param_match(args: Dict, entities: List[str]) -> Tuple[int, int, List[Dict], List[str], List[str]]:
    """
    检查args中的参数值是否在query实体列表中
    
    简化逻辑：
    - 只检查特定参数名（如 device_name, ip_address 等）
    - 参数值在 entities 列表中即为正确
    - 不再区分实体类型，避免字段名不匹配问题
    
    Args:
        args: action中的参数
        entities: 从query中提取的实体值列表
        
    Returns:
        (正确匹配数, 需要检查的参数总数, 不匹配的详情列表, 被检查的参数名列表, 正确的参数名列表)
    """
    correct_count = 0
    total_count = 0
    mismatch_details = []
    checked_params = []
    correct_params_list = []
    
    # 如果query中没有提取到实体，不进行检查
    if not entities:
        return 0, 0, [], [], []
    
    # 将实体列表转换为集合，方便查找
    entities_set = set(entities)
    
    for param_name, param_value in args.items():
        # 只检查特定的参数名
        if param_name.lower() not in QUERY_RELATED_PARAMS:
            continue
        
        # 将参数值转为字符串进行比较
        param_value_str = str(param_value)
        
        total_count += 1
        checked_params.append(param_name)
        
        # 检查参数值是否在实体列表中
        if param_value_str in entities_set:
            correct_count += 1
            correct_params_list.append(param_name)
        else:
            mismatch_details.append({
                'param_name': param_name,
                'param_value': param_value_str,
                'query_entities': entities,
                'error_type': 'not_in_query',
                'reason': f'参数值不在Query实体列表中'
            })
    
    return correct_count, total_count, mismatch_details, checked_params, correct_params_list


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
            for detail in file_result.get('calls_details', []):
                tool_name = detail.get('tool_name', '')
                action_valid = detail.get('action_valid', True)
                if not action_valid:
                    action_invalid_tools.add(tool_name)
            
            valid_calls_map[file_path]['_action_invalid_tools'] = action_invalid_tools
            
    except Exception as e:
        print(f"警告: 加载schema验证结果失败: {e}")
    
    return valid_calls_map


def calculate_query_param_accuracy(data_folders: List[str], 
                                    model_path: Optional[str] = None,
                                    entities_config_path: Optional[str] = None,
                                    schema_result_path: Optional[str] = None) -> Dict:
    """
    计算Query参数引用正确率
    
    Args:
        data_folders: 数据文件夹路径列表
        model_path: Qwen模型路径（可选，如果提供entities_config_path则不需要）
        entities_config_path: 实体配置文件路径（可选，JSON格式，键为文件夹名，值为实体列表）
        schema_result_path: schema验证结果文件路径（可选），用于计算联合指标
        
    Returns:
        包含统计结果的字典，包括每条数据的结果和总体结果
    """
    # 加载schema验证结果（如果提供）
    schema_valid_map = {}
    if schema_result_path:
        print(f"加载schema验证结果: {schema_result_path}")
        schema_valid_map = load_schema_validation_results(schema_result_path)
    
    # 确定使用哪种模式
    use_config_mode = entities_config_path is not None
    entities_config = {}
    model = None
    tokenizer = None
    
    if use_config_mode:
        # 配置文件模式：直接加载实体列表
        print("使用配置文件模式（跳过LLM提取）")
        entities_config = load_entities_from_config(entities_config_path)
    else:
        # LLM模式：加载模型进行实体提取
        if not model_path:
            raise ValueError("必须提供 model_path 或 entities_config_path 之一")
        print("使用LLM提取模式")
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
    
    # 统计变量 - 联合指标（仅schema有效的calls）
    schema_valid_total_params = 0
    schema_valid_correct_params = 0
    
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
                
                # 根据模式获取实体
                if use_config_mode:
                    # 配置文件模式：从配置中获取该文件夹的实体
                    folder_name = data_path.name
                    entities = entities_config.get(folder_name, [])
                    if not entities:
                        print(f"    警告: 配置文件中未找到文件夹 '{folder_name}' 的实体列表")
                    print(f"    从配置文件获取的实体: {entities}")
                else:
                    # LLM模式：使用模型提取实体
                    entities = extract_entities_from_query(query, model, tokenizer)
                    print(f"    LLM提取的实体: {entities}")
                
                # 检查response中的所有action参数
                per_step_details = []  # 记录每个step的检查详情
                
                if 'response' in data:
                    for step_dict in data['response']:
                        for step_key, step_value in step_dict.items():
                            if 'coa' in step_value:
                                coa_list = step_value['coa']
                                for coa_idx, coa_item in enumerate(coa_list):
                                    if 'action' in coa_item:
                                        action = coa_item['action']
                                        tool_name = action.get('name', '')
                                        args = action.get('args', {})
                                        
                                        # 检查参数匹配（新增返回 checked_params 和 correct_params_list）
                                        correct, total, mismatches, checked_params, correct_params_list = check_param_match(args, entities)
                                        
                                        # 记录该步骤的检查详情
                                        step_detail = {
                                            'step': step_key,
                                            'coa_index': coa_idx,
                                            'tool_name': tool_name,
                                            'checked_params': checked_params,
                                            'correct_params': correct_params_list
                                        }
                                        per_step_details.append(step_detail)
                                        
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
                    'per_step_details': per_step_details,  # 新增：每个step的检查详情
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
            'incorrect_params': total_params - correct_params
        }
    }
    
    # 如果启用联合指标，添加相关统计
    if schema_result_path:
        result['overall_schema_valid'] = {
            'Acc_param_query_schema_valid': schema_valid_accuracy,
            'total_params': schema_valid_total_params,
            'correct_params': schema_valid_correct_params,
            'incorrect_params': schema_valid_total_params - schema_valid_correct_params
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
                print(f"    - Step: {detail['step']}, 工具: {detail['tool_name']}")
                print(f"      参数名: {detail['param_name']}, 参数值: {detail['param_value']}")
                print(f"      Query实体: {detail.get('query_entities', [])}")
                print(f"      原因: {detail['reason']}")
    
    # 打印总体结果
    print("\n" + "="*60)
    print("总体统计结果")
    print("="*60)
    overall = result['overall']
    print(f"需检查参数总数 (N_query_param): {overall['total_params']}")
    print(f"正确引用参数数 (N_correct_query): {overall['correct_params']}")
    print(f"错误引用参数数: {overall['incorrect_params']}")
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
        print(f"Query参数引用正确率 (Schema有效): {schema_valid['Acc_param_query_schema_valid']:.4f} ({schema_valid['Acc_param_query_schema_valid']*100:.2f}%)")
    
    print("="*60)


if __name__ == "__main__":
    # 配置路径
    data_folders = [
        "/mnt/data/kw/ly/precision_index/data1",
        "/mnt/data/kw/ly/precision_index/data2",
        # 添加更多数据文件夹路径
    ]
    
    # Schema验证结果路径（用于计算联合指标）
    schema_result_path = "/mnt/data/kw/ly/precision_index/schema_validation_accuracy_result.json"
    
    # ========== 选择模式 ==========
    # 模式1: 使用配置文件模式（推荐，快速且稳定）
    # 直接提供每个文件夹的实体列表，跳过LLM提取
    entities_config_path = "entities_config.json"  # 实体配置文件路径
    result = calculate_query_param_accuracy(
        data_folders, 
        entities_config_path=entities_config_path,
        schema_result_path=schema_result_path
    )
    
    # 模式2: 使用LLM提取模式（较慢，可能出现解析错误）
    # 使用Qwen模型从query中提取实体
    # model_path = "/mnt/data/kw/models/Qwen/Qwen2.5-7B-Instruct"
    # result = calculate_query_param_accuracy(
    #     data_folders, 
    #     model_path=model_path,
    #     schema_result_path=schema_result_path
    # )
    
    # 打印结果
    print_results(result)
    
    # 保存结果到JSON文件
    output_file = "query_param_accuracy_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_file}")
