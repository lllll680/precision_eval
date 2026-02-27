#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
历史 Observation 参数引用正确率计算脚本

指标定义：
Acc_param_obs = N_correct_obs / N_obs_param

检查每个 step 的 action 参数是否正确引用了历史 observation 中的值
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_all_values_from_observation(obj: Any) -> Set[str]:
    """
    递归提取 observation 中的所有标量值
    
    Args:
        obj: observation 对象（可以是 dict, list, 或标量值）
        
    Returns:
        所有标量值的集合（转为字符串）
    """
    values = set()
    
    if obj is None:
        return values
    
    if isinstance(obj, dict):
        for v in obj.values():
            values.update(extract_all_values_from_observation(v))
    elif isinstance(obj, list):
        for item in obj:
            values.update(extract_all_values_from_observation(item))
    else:
        # 标量值：转为字符串存储
        str_val = str(obj).strip()
        if str_val and str_val.lower() not in ('none', 'null', ''):
            values.add(str_val)
    
    return values


def build_history_observation_values(response: List, current_step_index: int) -> Set[str]:
    """
    构建当前 step 之前所有 observation 的值集合
    
    Args:
        response: response 列表
        current_step_index: 当前 step 的索引（0-based）
        
    Returns:
        历史 observation 中所有值的集合
    """
    history_values = set()
    
    for i, step_item in enumerate(response):
        if i >= current_step_index:
            break
        
        # step_item 格式: {"stepN": {"cot": ..., "coa": [...]}}
        for step_key, step_data in step_item.items():
            if not step_key.startswith('step'):
                continue
            if isinstance(step_data, dict) and 'coa' in step_data:
                for coa_item in step_data['coa']:
                    if 'observation' in coa_item:
                        obs = coa_item['observation']
                        history_values.update(extract_all_values_from_observation(obs))
    
    return history_values


def build_future_observation_values(response: List, current_step_index: int) -> Set[str]:
    """
    构建当前 step 之后所有 observation 的值集合（用于检测未来引用）
    
    Args:
        response: response 列表
        current_step_index: 当前 step 的索引（0-based）
        
    Returns:
        未来 observation 中所有值的集合
    """
    future_values = set()
    
    for i, step_item in enumerate(response):
        if i <= current_step_index:
            continue
        
        # step_item 格式: {"stepN": {"cot": ..., "coa": [...]}}
        for step_key, step_data in step_item.items():
            if not step_key.startswith('step'):
                continue
            if isinstance(step_data, dict) and 'coa' in step_data:
                for coa_item in step_data['coa']:
                    if 'observation' in coa_item:
                        obs = coa_item['observation']
                        future_values.update(extract_all_values_from_observation(obs))
    
    return future_values


def value_in_set(param_value: str, value_set: Set[str]) -> bool:
    """
    检查参数值是否在值集合中（支持精确匹配和包含匹配）
    
    Args:
        param_value: 参数值
        value_set: 值集合
        
    Returns:
        是否匹配
    """
    param_str = str(param_value).strip()
    
    # 精确匹配
    if param_str in value_set:
        return True
    
    # 包含匹配：参数值是某个历史值的子串，或历史值是参数值的子串
    for hist_val in value_set:
        if len(param_str) >= 3 and len(hist_val) >= 3:  # 避免太短的匹配
            if param_str in hist_val or hist_val in param_str:
                return True
    
    return False


def is_value_in_query_entities(param_value: str, query_entities: Dict[str, List[str]]) -> bool:
    """
    检查参数值是否来源于 query 实体
    
    Args:
        param_value: 参数值
        query_entities: query 中提取的实体
        
    Returns:
        是否在 query 实体中
    """
    param_str = str(param_value).strip()
    
    for entity_type, values in query_entities.items():
        for v in values:
            if param_str == v or param_str in v or v in param_str:
                return True
    
    return False


# 常量白名单（这些值通常是工具默认参数，不需要从历史引用）
CONSTANT_WHITELIST = {
    # 常见数值
    '0', '1', '10', '100', '1000',
    # 布尔值
    'true', 'false', 'True', 'False',
    # 空值
    '', 'null', 'None',
}


def is_constant_value(param_value: str) -> bool:
    """
    检查参数值是否是常量
    
    Args:
        param_value: 参数值
        
    Returns:
        是否是常量
    """
    param_str = str(param_value).strip().lower()
    
    # 在白名单中
    if param_str in CONSTANT_WHITELIST or param_value in CONSTANT_WHITELIST:
        return True
    
    # 纯数字
    try:
        float(param_str)
        return True
    except ValueError:
        pass
    
    return False


def extract_json_robust(text: str) -> Optional[Dict]:
    """
    从文本中提取JSON（支持多种格式，包括Qwen模型的<think>标签）
    """
    original_text = text
    
    # 移除<think>标签及其内容（多种格式）
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<think>.*', '', text, flags=re.IGNORECASE)  # 处理未闭合的<think>
    
    # 移除其他可能的标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 移除markdown代码块标记
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # 尝试直接解析
    try:
        return json.loads(text.strip())
    except:
        pass
    
    # 查找JSON对象（更宽松的匹配）
    json_match = re.search(r'\{[^{}]*"(verified|from_context)"[^{}]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass
    
    # 尝试提取任何JSON对象
    json_match = re.search(r'\{.*?\}', text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(0))
            # 验证是否包含期望的字段
            if 'from_context' in result or 'verified' in result:
                return result
        except:
            pass
    
    # 尝试从原始文本中查找true/false关键词
    if 'true' in text.lower() or '"from_context": true' in original_text.lower():
        return {'from_context': True}
    elif 'false' in text.lower() or '"from_context": false' in original_text.lower():
        return {'from_context': False}
    
    return None


def extract_current_cot(response: List, step_index: int) -> str:
    """
    提取当前step的CoT文本
    
    Args:
        response: response列表
        step_index: 当前step索引
        
    Returns:
        当前step的CoT文本
    """
    if step_index >= len(response):
        return ""
    
    step_item = response[step_index]
    for step_key, step_data in step_item.items():
        if step_key.startswith('step') and isinstance(step_data, dict):
            return step_data.get('cot', '')
    
    return ""


def verify_param_from_context(
    param_value: str,
    history_obs_values: Set[str],
    current_cot: str,
    model,
    tokenizer,
    cache: Dict[str, bool] = None
) -> bool:
    """
    使用LLM验证参数值是否可以从历史observation或当前CoT中推断出来
    
    Args:
        param_value: 参数值
        history_obs_values: 历史observation值集合
        current_cot: 当前step的CoT文本
        model: LLM模型
        tokenizer: tokenizer
        cache: 缓存字典（可选）
        
    Returns:
        是否可以从上下文推断
    """
    # 构建缓存key
    history_str = '|'.join(sorted(list(history_obs_values)[:50]))  # 限制长度避免key过长
    cache_key = f"{history_str}||{current_cot[:200]}||{param_value}"
    
    # 检查缓存
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    
    # 构建简洁的prompt
    history_obs_str = ', '.join(list(history_obs_values)[:30]) if history_obs_values else "无"
    
    prompt = f"""判断参数值是否可以从给定上下文中合理推断出来。

历史观察值: {history_obs_str}
当前推理: {current_cot if current_cot else "无"}
参数值: {param_value}

要求：
1. 如果参数值在历史观察值中出现，返回true
2. 如果参数值可以从当前推理中提取或推断，返回true
3. 支持语义匹配（如"光模块"对应"optical_model"，"CPU"对应"cpu"）
4. 否则返回false

只输出JSON: {{"from_context": true/false}}

JSON输出："""

    # 调用模型
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=128,  # 减少token数提高速度
        temperature=0.1,
        top_p=0.9,
        do_sample=True
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # 解析结果
    result = extract_json_robust(response)
    
    if result and 'from_context' in result:
        verified = bool(result['from_context'])
    else:
        # 默认保守：如果无法解析，认为不是幻觉（避免误判）
        print(f"\n警告: 无法解析LLM响应")
        print(f"完整响应: {response}")
        print(f"参数值: {param_value}")
        print(f"提取结果: {result}")
        print("-" * 60)
        verified = True
    
    # 缓存结果
    if cache is not None:
        cache[cache_key] = verified
    
    return verified


def check_obs_param_match(
    args: Dict,
    query_checked_params: List[str],
    history_obs_values: Set[str],
    future_obs_values: Set[str],
    current_cot: str,
    step_index: int,
    model,
    tokenizer,
    llm_cache: Dict[str, bool]
) -> Tuple[int, int, List[Dict]]:
    """
    检查 action 参数是否正确引用了历史 observation
    
    Args:
        args: action 的参数字典
        query_checked_params: query_param_accuracy 已检查的参数名列表（需排除）
        history_obs_values: 历史 observation 值集合
        future_obs_values: 未来 observation 值集合
        current_cot: 当前step的CoT文本
        step_index: 当前 step 索引
        model: LLM模型
        tokenizer: tokenizer
        llm_cache: LLM判断结果缓存
        
    Returns:
        (correct_count, total_count, error_details)
    """
    correct_count = 0
    total_count = 0
    error_details = []
    
    for param_name, param_value in args.items():
        # 跳过空值
        if param_value is None or str(param_value).strip() == '':
            continue
        
        param_str = str(param_value).strip()
        
        # 跳过常量值
        if is_constant_value(param_str):
            continue
        
        # 跳过已被 query_param_accuracy 检查的参数（基于参数名）
        if param_name in query_checked_params:
            continue
        
        # 到这里，是 observation 参数，需要检查
        total_count += 1
        
        # 第一步：检查是否在历史 observation 中（快速匹配）
        if value_in_set(param_str, history_obs_values):
            correct_count += 1
            continue
        
        # 第二步：使用LLM进行语义判断
        verified = verify_param_from_context(
            param_str, history_obs_values, current_cot, model, tokenizer, llm_cache
        )
        
        if verified:
            correct_count += 1
        else:
            # 错误：确定错误类型（合并no_history到hallucination）
            if value_in_set(param_str, future_obs_values):
                error_type = 'future_reference'
                reason = '参数值存在于未来的observation中，可能是提前引用'
            else:
                error_type = 'hallucination'
                if step_index == 0:
                    reason = '第一个step，参数值既不来自query也无法从上下文推断'
                else:
                    reason = '参数值无法从历史observation或当前CoT中推断，可能是幻觉'
            
            error_details.append({
                'param_name': param_name,
                'param_value': param_str,
                'error_type': error_type,
                'reason': reason
            })
    
    return correct_count, total_count, error_details


def load_query_param_results(query_result_path: str) -> Dict[str, Dict[str, Any]]:
    """
    加载 query 参数引用正确率的结果，获取每个文件的实体提取结果和已检查参数
    
    Args:
        query_result_path: query_param_accuracy_result.json 路径
        
    Returns:
        {file_path: {
            'extracted_entities': {...},
            'query': '...',
            'checked_params_map': {(step, coa_index, tool_name): [checked_params]}
        }}
    """
    result_map = {}
    
    try:
        with open(query_result_path, 'r', encoding='utf-8') as f:
            query_result = json.load(f)
        
        for file_result in query_result.get('per_file_results', []):
            file_path = file_result['file']
            
            # 构建已检查参数映射: (step, coa_index, tool_name) -> checked_params
            checked_params_map = {}
            for step_detail in file_result.get('per_step_details', []):
                step = step_detail.get('step', '')
                coa_index = step_detail.get('coa_index', 0)
                tool_name = step_detail.get('tool_name', '')
                checked_params = step_detail.get('checked_params', [])
                
                key = (step, coa_index, tool_name)
                checked_params_map[key] = checked_params
            
            result_map[file_path] = {
                'extracted_entities': file_result.get('extracted_entities', {}),
                'query': file_result.get('query', ''),
                'checked_params_map': checked_params_map
            }
    except Exception as e:
        print(f"警告: 加载query参数结果失败: {e}")
    
    return result_map


def load_schema_validation_results(schema_result_path: str) -> Dict[str, Dict[str, Any]]:
    """
    加载 schema 验证结果，用于联合指标计算
    
    Args:
        schema_result_path: schema_validation_accuracy_result.json 路径
        
    Returns:
        {file_path: {'_action_invalid_tools': set(), '_obs_invalid_tools': set()}}
    """
    valid_calls_map = {}
    
    try:
        with open(schema_result_path, 'r', encoding='utf-8') as f:
            schema_result = json.load(f)
        
        for file_result in schema_result.get('per_file_results', []):
            file_path = file_result['file']
            
            action_invalid_tools = set()
            obs_invalid_tools = set()
            
            for detail in file_result.get('calls_details', []):
                tool_name = detail.get('tool_name', '')
                action_valid = detail.get('action_valid', True)
                obs_valid = detail.get('observation_valid', True)
                
                if not action_valid:
                    action_invalid_tools.add(tool_name)
                if not obs_valid:
                    obs_invalid_tools.add(tool_name)
            
            valid_calls_map[file_path] = {
                '_action_invalid_tools': action_invalid_tools,
                '_obs_invalid_tools': obs_invalid_tools
            }
            
    except Exception as e:
        print(f"警告: 加载schema验证结果失败: {e}")
    
    return valid_calls_map


def calculate_obs_param_accuracy(
    data_folders: List[str],
    query_result_path: str,
    model_path: str,
    schema_result_path: Optional[str] = None,
    exclude_last_steps: int = 0
) -> Dict:
    """
    计算历史 Observation 参数引用正确率
    
    Args:
        data_folders: 数据文件夹路径列表
        query_result_path: query_param_accuracy_result.json 路径
        model_path: 本地LLM模型路径
        schema_result_path: schema 验证结果路径（可选，用于联合指标）
        exclude_last_steps: 排除最后 N 个 steps（默认 0 不排除，设为 2 排除最后两个总结 step）
        
    Returns:
        包含统计结果的字典
    """
    # 加载 query 参数结果（复用实体提取）
    print(f"加载query参数结果: {query_result_path}")
    query_results = load_query_param_results(query_result_path)
    
    # 加载本地LLM模型
    print(f"正在加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    print("模型加载完成")
    
    # LLM判断结果缓存
    llm_cache = {}
    
    # 加载 schema 验证结果（可选）
    schema_valid_map = {}
    if schema_result_path:
        print(f"加载schema验证结果: {schema_result_path}")
        schema_valid_map = load_schema_validation_results(schema_result_path)
    
    # 统计变量
    per_file_results = []
    total_obs_params = 0
    correct_obs_params = 0
    
    # 错误类型统计（合并no_history到hallucination）
    hallucination_errors = 0
    future_reference_errors = 0
    
    # 联合指标统计
    schema_valid_total_params = 0
    schema_valid_correct_params = 0
    schema_valid_hallucination = 0
    schema_valid_future_ref = 0
    
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
            
            file_path_str = str(json_file)
            
            # 获取该文件的 query 实体提取结果
            if file_path_str not in query_results:
                print(f"  跳过文件: {json_file.name} (未在query结果中找到)")
                continue
            
            query_info = query_results[file_path_str]
            query_entities = query_info['extracted_entities']
            query = query_info['query']
            checked_params_map = query_info.get('checked_params_map', {})
            
            print(f"  处理文件: {json_file.name}")
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                response = data.get('response', [])
                
                # 计算需要处理的 steps 数量（排除最后 N 个）
                total_steps = len(response)
                steps_to_process = max(0, total_steps - exclude_last_steps)
                
                if exclude_last_steps > 0 and total_steps > steps_to_process:
                    print(f"    排除最后 {exclude_last_steps} 个 steps（共 {total_steps} 个，处理前 {steps_to_process} 个）")
                
                # 文件级统计
                file_total_params = 0
                file_correct_params = 0
                file_error_details = []
                
                # 文件级联合指标
                file_schema_valid_total = 0
                file_schema_valid_correct = 0
                
                # 获取该文件的 schema 验证信息
                file_schema_info = schema_valid_map.get(file_path_str, {})
                
                # 遍历每个 step（response 是列表），排除最后 N 个
                for step_index, step_item in enumerate(response[:steps_to_process]):
                    # step_item 格式: {"stepN": {"cot": ..., "coa": [...]}}
                    for step_key, step_data in step_item.items():
                        if not step_key.startswith('step'):
                            continue
                        
                        if not isinstance(step_data, dict) or 'coa' not in step_data:
                            continue
                        
                        # 遍历 coa 中的每个 action-observation 对
                        for coa_idx, coa_item in enumerate(step_data['coa']):
                            action = coa_item.get('action', {})
                            if not isinstance(action, dict):
                                continue
                            
                            tool_name = action.get('name', '')  # 注意：用 'name' 而非 'tool_name'
                            args = action.get('args', {})
                            
                            if not args or not isinstance(args, dict):
                                continue
                            
                            # 获取 query_param_accuracy 已检查的参数名列表
                            checked_key = (step_key, coa_idx, tool_name)
                            query_checked_params = checked_params_map.get(checked_key, [])
                            
                            # 构建历史和未来 observation 值集合
                            history_obs_values = build_history_observation_values(response, step_index)
                            future_obs_values = build_future_observation_values(response, step_index)
                            
                            # 提取当前step的CoT
                            current_cot = extract_current_cot(response, step_index)
                            
                            # 检查参数匹配（传入已检查参数名列表、CoT和LLM）
                            correct, total, errors = check_obs_param_match(
                                args, query_checked_params, history_obs_values, future_obs_values,
                                current_cot, step_index, model, tokenizer, llm_cache
                            )
                            
                            # 独立指标统计
                            file_total_params += total
                            file_correct_params += correct
                            total_obs_params += total
                            correct_obs_params += correct
                            
                            # 判断该 call 的 action 是否 schema 有效
                            action_invalid_tools = file_schema_info.get('_action_invalid_tools', set())
                            is_action_valid = tool_name not in action_invalid_tools
                            
                            # 联合指标统计（仅 action 有效的 calls）
                            if schema_result_path and is_action_valid:
                                file_schema_valid_total += total
                                file_schema_valid_correct += correct
                                schema_valid_total_params += total
                                schema_valid_correct_params += correct
                            
                            # 记录错误详情
                            for error in errors:
                                error['tool_name'] = tool_name
                                error['step'] = step_key
                                error['action_valid'] = is_action_valid
                                file_error_details.append(error)
                                
                                # 统计错误类型
                                error_type = error['error_type']
                                if error_type == 'hallucination':
                                    hallucination_errors += 1
                                    if schema_result_path and is_action_valid:
                                        schema_valid_hallucination += 1
                                elif error_type == 'future_reference':
                                    future_reference_errors += 1
                                    if schema_result_path and is_action_valid:
                                        schema_valid_future_ref += 1
                
                # 计算文件级准确率
                file_accuracy = file_correct_params / file_total_params if file_total_params > 0 else 0.0
                file_schema_valid_accuracy = file_schema_valid_correct / file_schema_valid_total if file_schema_valid_total > 0 else 0.0
                
                # 记录文件结果
                file_result = {
                    'file': file_path_str,
                    'query': query,
                    'extracted_entities': query_entities,
                    'Acc_param_obs': file_accuracy,
                    'total_obs_params': file_total_params,
                    'correct_obs_params': file_correct_params,
                    'incorrect_obs_params': file_total_params - file_correct_params,
                    'error_details': file_error_details
                }
                
                # 添加联合指标
                if schema_result_path:
                    file_result['schema_valid_metrics'] = {
                        'Acc_param_obs_action_valid': file_schema_valid_accuracy,
                        'total_params': file_schema_valid_total,
                        'correct_params': file_schema_valid_correct,
                        'incorrect_params': file_schema_valid_total - file_schema_valid_correct
                    }
                
                per_file_results.append(file_result)
                
            except Exception as e:
                print(f"    处理文件 {json_file} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 计算总体准确率
    overall_accuracy = correct_obs_params / total_obs_params if total_obs_params > 0 else 0.0
    schema_valid_accuracy = schema_valid_correct_params / schema_valid_total_params if schema_valid_total_params > 0 else 0.0
    
    # 构建结果
    result = {
        'per_file_results': per_file_results,
        'overall': {
            'Acc_param_obs': overall_accuracy,
            'total_obs_params': total_obs_params,
            'correct_obs_params': correct_obs_params,
            'incorrect_obs_params': total_obs_params - correct_obs_params,
            'error_breakdown': {
                'hallucination': hallucination_errors,
                'future_reference': future_reference_errors
            }
        }
    }
    
    # 添加联合指标
    if schema_result_path:
        result['overall_schema_valid'] = {
            'Acc_param_obs_action_valid': schema_valid_accuracy,
            'total_params': schema_valid_total_params,
            'correct_params': schema_valid_correct_params,
            'incorrect_params': schema_valid_total_params - schema_valid_correct_params,
            'error_breakdown': {
                'hallucination': schema_valid_hallucination,
                'future_reference': schema_valid_future_ref
            }
        }
    
    return result


def print_results(result: Dict):
    """
    打印统计结果
    """
    print("\n" + "="*60)
    print("历史Observation参数引用正确率统计结果")
    print("="*60)
    
    # 打印每条数据的结果
    print("\n每条数据的结果:")
    print("-"*60)
    for file_result in result['per_file_results']:
        print(f"\n文件: {file_result['file']}")
        print(f"  Query: {file_result['query']}")
        print(f"  需检查参数数: {file_result['total_obs_params']}")
        print(f"  正确参数数: {file_result['correct_obs_params']}")
        print(f"  错误参数数: {file_result['incorrect_obs_params']}")
        print(f"  Acc_param_obs: {file_result['Acc_param_obs']:.4f} ({file_result['Acc_param_obs']*100:.2f}%)")
        
        # 显示联合指标
        if 'schema_valid_metrics' in file_result:
            svm = file_result['schema_valid_metrics']
            print(f"  [联合指标-仅Action有效] 检查参数数: {svm['total_params']}, 正确: {svm['correct_params']}, "
                  f"Acc: {svm['Acc_param_obs_action_valid']:.4f} ({svm['Acc_param_obs_action_valid']*100:.2f}%)")
        
        if file_result['error_details']:
            print(f"  错误详情:")
            for detail in file_result['error_details']:
                error_type_cn = {
                    'hallucination': '幻觉',
                    'future_reference': '未来引用'
                }.get(detail['error_type'], detail['error_type'])
                
                print(f"    - Step: {detail['step']}, 工具: {detail['tool_name']}")
                print(f"      错误类型: {error_type_cn}")
                print(f"      参数名: {detail['param_name']}, 参数值: {detail['param_value']}")
                print(f"      原因: {detail['reason']}")
    
    # 打印总体结果
    print("\n" + "="*60)
    print("总体统计结果")
    print("="*60)
    overall = result['overall']
    print(f"需检查参数总数 (N_obs_param): {overall['total_obs_params']}")
    print(f"正确引用参数数 (N_correct_obs): {overall['correct_obs_params']}")
    print(f"错误引用参数数: {overall['incorrect_obs_params']}")
    
    # 打印错误类型分布
    breakdown = overall['error_breakdown']
    print(f"  - 幻觉 (hallucination): {breakdown['hallucination']}")
    print(f"  - 未来引用 (future_reference): {breakdown['future_reference']}")
    
    print(f"Observation参数引用正确率 (Acc_param_obs): {overall['Acc_param_obs']:.4f} ({overall['Acc_param_obs']*100:.2f}%)")
    
    # 打印联合指标
    if 'overall_schema_valid' in result:
        print("\n" + "-"*60)
        print("联合指标 - 仅Action有效的Calls")
        print("-"*60)
        schema_valid = result['overall_schema_valid']
        print(f"需检查参数总数 (Action有效): {schema_valid['total_params']}")
        print(f"正确引用参数数: {schema_valid['correct_params']}")
        print(f"错误引用参数数: {schema_valid['incorrect_params']}")
        
        breakdown = schema_valid['error_breakdown']
        print(f"  - 幻觉 (hallucination): {breakdown['hallucination']}")
        print(f"  - 未来引用 (future_reference): {breakdown['future_reference']}")
        
        print(f"Observation参数引用正确率 (Action有效): {schema_valid['Acc_param_obs_action_valid']:.4f} ({schema_valid['Acc_param_obs_action_valid']*100:.2f}%)")
    
    print("="*60)


if __name__ == "__main__":
    # 配置路径
    data_folders = [
        "/mnt/data/kw/ly/precision_index/data1",
        "/mnt/data/kw/ly/precision_index/data2",
    ]
    
    # Query参数结果路径（复用实体提取）
    query_result_path = "/mnt/data/kw/ly/precision_index/query_param_accuracy_result.json"
    
    # 本地LLM模型路径
    model_path = "/mnt/data/kw/models/Qwen/Qwen2.5-7B-Instruct"
    
    # Schema验证结果路径（用于联合指标）
    schema_result_path = "/mnt/data/kw/ly/precision_index/schema_validation_accuracy_result.json"
    
    # 排除最后 N 个 steps（设为 2 排除最后两个总结 step，设为 0 不排除）
    exclude_last_steps = 2
    
    # 计算Observation参数引用正确率
    result = calculate_obs_param_accuracy(
        data_folders, query_result_path, model_path, schema_result_path, exclude_last_steps
    )
    
    # 打印结果
    print_results(result)
    
    # 保存结果到JSON文件
    output_file = "obs_param_accuracy_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_file}")
