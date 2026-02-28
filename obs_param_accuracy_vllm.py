#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
历史 Observation 参数引用正确率计算脚本 - vLLM优化版本

性能优化：
1. 使用 vLLM 替代 transformers，支持高效批量推理
2. 自动利用多GPU（tensor_parallel_size）
3. PagedAttention 优化显存使用
4. 连续批处理提高吞吐量

预期性能提升：10-20倍
GPU利用率：30% → 90%+

使用方法：
    CUDA_VISIBLE_DEVICES=0,1,2,3 python obs_param_accuracy_vllm.py
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple, Optional
from vllm import LLM, SamplingParams


def extract_all_values_from_observation(obj: Any) -> Set[str]:
    """
    递归提取 observation 中的所有标量值
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
        str_val = str(obj).strip()
        if str_val and str_val.lower() not in ('none', 'null', ''):
            values.add(str_val)
    
    return values


def build_history_observation_values(response: List, current_step_index: int) -> Set[str]:
    """
    构建历史 observation 值集合（当前 step 之前的所有 observation）
    """
    history_values = set()
    
    for step_idx in range(current_step_index):
        if step_idx >= len(response):
            break
        
        step_item = response[step_idx]
        for step_key, step_data in step_item.items():
            if not step_key.startswith('step'):
                continue
            
            if 'coa' not in step_data:
                continue
            
            coa_list = step_data['coa']
            for coa_item in coa_list:
                if 'observation' in coa_item:
                    obs = coa_item['observation']
                    history_values.update(extract_all_values_from_observation(obs))
    
    return history_values


def build_future_observation_values(response: List, current_step_index: int) -> Set[str]:
    """
    构建未来 observation 值集合（当前 step 之后的所有 observation）
    """
    future_values = set()
    
    for step_idx in range(current_step_index + 1, len(response)):
        step_item = response[step_idx]
        for step_key, step_data in step_item.items():
            if not step_key.startswith('step'):
                continue
            
            if 'coa' not in step_data:
                continue
            
            coa_list = step_data['coa']
            for coa_item in coa_list:
                if 'observation' in coa_item:
                    obs = coa_item['observation']
                    future_values.update(extract_all_values_from_observation(obs))
    
    return future_values


def is_constant_value(value: str) -> bool:
    """
    判断是否为常量值
    """
    constant_patterns = [
        r'^\d+$',
        r'^true$', r'^false$',
        r'^yes$', r'^no$',
        r'^on$', r'^off$',
        r'^enabled?$', r'^disabled?$',
        r'^up$', r'^down$',
        r'^active$', r'^inactive$',
        r'^running$', r'^stopped$',
    ]
    
    value_lower = value.lower().strip()
    
    for pattern in constant_patterns:
        if re.match(pattern, value_lower, re.IGNORECASE):
            return True
    
    return False


def value_in_set(value: str, value_set: Set[str]) -> bool:
    """
    检查值是否在集合中（支持部分匹配）
    """
    value = value.strip()
    
    if value in value_set:
        return True
    
    for v in value_set:
        if value in v or v in value:
            return True
    
    return False


def get_current_cot(response: List, step_index: int) -> str:
    """
    获取当前step的CoT文本
    """
    if step_index >= len(response):
        return ""
    
    step_item = response[step_index]
    for step_key, step_data in step_item.items():
        if step_key.startswith('step') and isinstance(step_data, dict):
            return step_data.get('cot', '')
    
    return ""


def extract_json_robust(text: str) -> Optional[Dict]:
    """
    从文本中提取JSON（支持多种格式）
    """
    original_text = text
    
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<think>.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    try:
        return json.loads(text.strip())
    except:
        pass
    
    json_match = re.search(r'\{[^{}]*"(verified|from_context)"[^{}]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass
    
    json_match = re.search(r'\{.*?\}', text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(0))
            if 'from_context' in result or 'verified' in result:
                return result
        except:
            pass
    
    if 'true' in text.lower() or '"from_context": true' in original_text.lower():
        return {'from_context': True}
    elif 'false' in text.lower() or '"from_context": false' in original_text.lower():
        return {'from_context': False}
    
    return None


def verify_params_batch_vllm(
    param_contexts: List[Tuple[str, Set[str], str]],
    llm: LLM,
    cache: Dict[str, bool] = None
) -> List[bool]:
    """
    使用vLLM批量验证参数
    
    Args:
        param_contexts: [(param_value, history_obs_values, current_cot), ...]
        llm: vLLM模型实例
        cache: 缓存字典
        
    Returns:
        验证结果列表
    """
    if not param_contexts:
        return []
    
    results = []
    uncached_indices = []
    uncached_prompts = []
    cache_keys = []
    
    for idx, (param_value, history_obs_values, current_cot) in enumerate(param_contexts):
        history_str = '|'.join(sorted(list(history_obs_values)[:50]))
        cache_key = f"{history_str}||{current_cot[:200]}||{param_value}"
        cache_keys.append(cache_key)
        
        if cache is not None and cache_key in cache:
            results.append(cache[cache_key])
        else:
            results.append(None)
            uncached_indices.append(idx)
            
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
            uncached_prompts.append(prompt)
    
    if uncached_prompts:
        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=128
        )
        
        outputs = llm.generate(uncached_prompts, sampling_params)
        
        for i, output in enumerate(outputs):
            idx = uncached_indices[i]
            response_text = output.outputs[0].text
            
            result = extract_json_robust(response_text)
            verified = bool(result.get('from_context', True)) if result else True
            
            results[idx] = verified
            
            if cache is not None:
                cache[cache_keys[idx]] = verified
    
    return results


def load_query_param_results(query_result_path: str) -> Dict:
    """
    加载 query_param_accuracy 的结果
    """
    with open(query_result_path, 'r', encoding='utf-8') as f:
        result = json.load(f)
    
    file_map = {}
    for file_result in result.get('per_file_results', []):
        file_path = file_result['file']
        file_map[file_path] = {
            'extracted_entities': file_result.get('extracted_entities', []),
            'query': file_result.get('query', ''),
            'checked_params_map': file_result.get('checked_params_map', {})
        }
    
    return file_map


def load_schema_validation_results(schema_result_path: str) -> Dict:
    """
    加载 schema_validation_accuracy 的结果
    """
    try:
        with open(schema_result_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        file_map = {}
        for file_result in result.get('per_file_results', []):
            file_path = file_result['file']
            file_map[file_path] = {
                'calls_details': file_result.get('calls_details', [])
            }
        
        return file_map
    except:
        return {}


def collect_params_to_verify(
    data_folders: List[str],
    query_results: Dict,
    schema_valid_map: Dict,
    exclude_last_steps: int = 0
) -> Tuple[List[Tuple], Dict]:
    """
    收集所有需要验证的参数（第一阶段）
    
    Returns:
        (params_to_verify, metadata)
        params_to_verify: [(param_value, history_obs, cot, file_path, step_key, param_name, step_index, future_obs), ...]
        metadata: 用于后续构建结果的元数据
    """
    params_to_verify = []
    metadata = {
        'file_info': {},
    }
    
    for folder in data_folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"警告: 文件夹不存在 {folder}")
            continue
        
        print(f"\n收集参数: {folder}")
        
        for json_file in sorted(folder_path.glob("*.json")):
            if json_file.name in ['question_info.json', 'batch_summary.json']:
                continue
            
            file_path_str = str(json_file)
            
            if file_path_str not in query_results:
                continue
            
            query_info = query_results[file_path_str]
            checked_params_map = query_info.get('checked_params_map', {})
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                response = data.get('response', [])
                total_steps = len(response)
                steps_to_process = max(0, total_steps - exclude_last_steps)
                
                file_schema_info = schema_valid_map.get(file_path_str, {})
                
                file_params_indices = []
                
                for step_index, step_item in enumerate(response[:steps_to_process]):
                    for step_key, step_data in step_item.items():
                        if not step_key.startswith('step'):
                            continue
                        
                        if 'coa' not in step_data:
                            continue
                        
                        coa_list = step_data['coa']
                        
                        history_obs_values = build_history_observation_values(response, step_index)
                        future_obs_values = build_future_observation_values(response, step_index)
                        current_cot = get_current_cot(response, step_index)
                        
                        query_checked_params = checked_params_map.get(step_key, [])
                        
                        for coa_item in coa_list:
                            if 'action' not in coa_item:
                                continue
                            
                            action = coa_item['action']
                            args = action.get('arguments', {})
                            
                            for param_name, param_value in args.items():
                                if param_value is None or str(param_value).strip() == '':
                                    continue
                                
                                param_str = str(param_value).strip()
                                
                                if is_constant_value(param_str):
                                    continue
                                
                                if param_name in query_checked_params:
                                    continue
                                
                                param_idx = len(params_to_verify)
                                file_params_indices.append(param_idx)
                                
                                params_to_verify.append((
                                    param_str,
                                    history_obs_values,
                                    current_cot,
                                    file_path_str,
                                    step_key,
                                    param_name,
                                    step_index,
                                    future_obs_values
                                ))
                
                metadata['file_info'][file_path_str] = {
                    'params_indices': file_params_indices,
                    'response': response,
                    'steps_to_process': steps_to_process
                }
                
            except Exception as e:
                print(f"  处理文件 {json_file} 时出错: {e}")
                continue
    
    return params_to_verify, metadata


def calculate_obs_param_accuracy(
    data_folders: List[str],
    query_result_path: str,
    model_path: str,
    schema_result_path: Optional[str] = None,
    exclude_last_steps: int = 0,
    tensor_parallel_size: int = 4,
    gpu_memory_utilization: float = 0.9
) -> Dict:
    """
    计算 Observation 参数引用正确率（vLLM优化版本）
    
    Args:
        data_folders: 数据文件夹路径列表
        query_result_path: query参数结果文件路径
        model_path: 模型路径
        schema_result_path: schema验证结果路径（可选）
        exclude_last_steps: 排除最后N个steps
        tensor_parallel_size: 使用的GPU数量
        gpu_memory_utilization: GPU显存利用率
    """
    print(f"加载query参数结果: {query_result_path}")
    query_results = load_query_param_results(query_result_path)
    
    print(f"\n正在加载vLLM模型: {model_path}")
    print(f"  - Tensor Parallel Size: {tensor_parallel_size}")
    print(f"  - GPU Memory Utilization: {gpu_memory_utilization}")
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=4096
    )
    print("✅ vLLM模型加载完成")
    
    llm_cache = {}
    
    schema_valid_map = {}
    if schema_result_path:
        print(f"加载schema验证结果: {schema_result_path}")
        schema_valid_map = load_schema_validation_results(schema_result_path)
    
    print("\n" + "="*60)
    print("阶段1：收集所有需要验证的参数")
    print("="*60)
    
    params_to_verify, metadata = collect_params_to_verify(
        data_folders, query_results, schema_valid_map, exclude_last_steps
    )
    
    print(f"\n✅ 收集完成，共 {len(params_to_verify)} 个参数需要验证")
    
    print("\n" + "="*60)
    print("阶段2：批量验证参数（vLLM加速）")
    print("="*60)
    
    param_contexts = [(p[0], p[1], p[2]) for p in params_to_verify]
    verification_results = verify_params_batch_vllm(param_contexts, llm, llm_cache)
    
    print(f"✅ 批量验证完成")
    
    print("\n" + "="*60)
    print("阶段3：构建结果")
    print("="*60)
    
    per_file_results = []
    total_obs_params = 0
    correct_obs_params = 0
    hallucination_errors = 0
    future_reference_errors = 0
    
    for file_path_str, file_info in metadata['file_info'].items():
        file_total_params = 0
        file_correct_params = 0
        file_error_details = []
        
        for param_idx in file_info['params_indices']:
            param_str, history_obs, cot, _, step_key, param_name, step_index, future_obs = params_to_verify[param_idx]
            verified = verification_results[param_idx]
            
            file_total_params += 1
            total_obs_params += 1
            
            if verified:
                file_correct_params += 1
                correct_obs_params += 1
            else:
                if value_in_set(param_str, future_obs):
                    error_type = 'future_reference'
                    reason = '参数值存在于未来的observation中，可能是提前引用'
                    future_reference_errors += 1
                else:
                    error_type = 'hallucination'
                    if step_index == 0:
                        reason = '第一个step，参数值既不来自query也无法从上下文推断'
                    else:
                        reason = '参数值无法从历史observation或当前CoT中推断，可能是幻觉'
                    hallucination_errors += 1
                
                file_error_details.append({
                    'step': step_key,
                    'param_name': param_name,
                    'param_value': param_str,
                    'error_type': error_type,
                    'reason': reason
                })
        
        file_accuracy = file_correct_params / file_total_params if file_total_params > 0 else 0.0
        
        per_file_results.append({
            'file': file_path_str,
            'total_obs_params': file_total_params,
            'correct_obs_params': file_correct_params,
            'incorrect_obs_params': file_total_params - file_correct_params,
            'Acc_param_obs': file_accuracy,
            'error_details': file_error_details[:10]
        })
    
    overall_accuracy = correct_obs_params / total_obs_params if total_obs_params > 0 else 0.0
    
    result = {
        'per_file_results': per_file_results,
        'overall': {
            'total_obs_params': total_obs_params,
            'correct_obs_params': correct_obs_params,
            'incorrect_obs_params': total_obs_params - correct_obs_params,
            'Acc_param_obs': overall_accuracy,
            'error_breakdown': {
                'hallucination': hallucination_errors,
                'future_reference': future_reference_errors
            }
        }
    }
    
    return result


def print_results(result: Dict):
    """打印统计结果"""
    print("\n" + "="*60)
    print("每条数据的结果")
    print("="*60)
    
    for file_result in result['per_file_results']:
        print(f"\n文件: {file_result['file']}")
        print(f"  需检查参数数: {file_result['total_obs_params']}")
        print(f"  正确参数数: {file_result['correct_obs_params']}")
        print(f"  错误参数数: {file_result['incorrect_obs_params']}")
        print(f"  Acc_param_obs: {file_result['Acc_param_obs']:.4f} ({file_result['Acc_param_obs']*100:.2f}%)")
        
        if file_result['error_details']:
            print(f"  错误详情:")
            for detail in file_result['error_details']:
                error_type_cn = {
                    'hallucination': '幻觉',
                    'future_reference': '未来引用'
                }.get(detail['error_type'], detail['error_type'])
                
                print(f"    - Step: {detail['step']}, 参数: {detail['param_name']}")
                print(f"      错误类型: {error_type_cn}, 值: {detail['param_value']}")
    
    print("\n" + "="*60)
    print("总体统计结果")
    print("="*60)
    overall = result['overall']
    print(f"需检查参数总数: {overall['total_obs_params']}")
    print(f"正确引用参数数: {overall['correct_obs_params']}")
    print(f"错误引用参数数: {overall['incorrect_obs_params']}")
    
    breakdown = overall['error_breakdown']
    print(f"  - 幻觉: {breakdown['hallucination']}")
    print(f"  - 未来引用: {breakdown['future_reference']}")
    
    print(f"Observation参数引用正确率: {overall['Acc_param_obs']:.4f} ({overall['Acc_param_obs']*100:.2f}%)")
    print("="*60)


if __name__ == "__main__":
    data_folders = [
        "/mnt/data/kw/ly/precision_index/data1",
        "/mnt/data/kw/ly/precision_index/data2",
    ]
    
    query_result_path = "/mnt/data/kw/ly/precision_index/query_param_accuracy_result.json"
    model_path = "/mnt/data/kw/models/Qwen/Qwen2.5-7B-Instruct"
    schema_result_path = "/mnt/data/kw/ly/precision_index/schema_validation_accuracy_result.json"
    
    exclude_last_steps = 2
    
    tensor_parallel_size = 4
    gpu_memory_utilization = 0.9
    
    result = calculate_obs_param_accuracy(
        data_folders,
        query_result_path,
        model_path,
        schema_result_path,
        exclude_last_steps,
        tensor_parallel_size,
        gpu_memory_utilization
    )
    
    print_results(result)
    
    output_file = "obs_param_accuracy_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_file}")
