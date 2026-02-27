#!/usr/bin/env python3
"""
标准化 tool.txt 文件，修复常见的 JSON 格式问题

功能：
1. 读取原始 tool.txt
2. 尝试解析每个工具的 Parameters 和 Output
3. 修复常见格式问题（中文标点、括号不匹配、关键字错误等）
4. 生成标准化的 tool_normalized.txt

使用方法：
    python normalize_tool_schema.py
"""

import json
import re
import ast
from pathlib import Path
from typing import Dict, Optional, Any, List


def fix_chinese_punctuation(s: str) -> str:
    """替换中文标点为英文标点"""
    replacements = {
        '，': ',',
        '：': ':',
        '；': ';',
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        '（': '(',
        '）': ')',
        '【': '[',
        '】': ']',
        '《': '<',
        '》': '>',
    }
    for cn, en in replacements.items():
        s = s.replace(cn, en)
    return s


def fix_properties_closure(s: str) -> str:
    """
    修复 properties 对象缺少闭合括号的问题
    
    原始问题：
    {'properties': {'key1': {...}}, 'key2': {...}, 'required': [...]}
    
    应该修复为：
    {'properties': {'key1': {...}, 'key2': {...}}, 'required': [...]}
    """
    # 查找 "properties": { ... 的位置
    props_match = re.search(r'["\']properties["\']\s*:\s*\{', s)
    if not props_match:
        return s
    
    props_start = props_match.end() - 1  # { 的位置
    
    # 从 properties 开始，找到对应的闭合 }
    # 同时检测是否遇到顶层关键字
    depth = 0
    in_string = False
    escape = False
    i = props_start
    
    top_level_keys = ['required', 'type', 'additionalProperties', 'description']
    
    # 记录最后一个可能的闭合位置（在遇到顶层关键字之前）
    last_valid_close = -1
    
    while i < len(s):
        c = s[i]
        
        if escape:
            escape = False
            i += 1
            continue
        
        if c == '\\':
            escape = True
            i += 1
            continue
        
        if c in ('"', "'") and not in_string:
            in_string = c
        elif c == in_string:
            in_string = False
        elif not in_string:
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    # properties 正常闭合
                    return s
                elif depth == 1:
                    # 记录 properties 内部属性的闭合位置
                    last_valid_close = i
            elif c == ',' and depth == 1:
                # 在 properties 的直接子级，检查后面是否跟着顶层关键字
                remaining = s[i+1:].lstrip()
                for key in top_level_keys:
                    if remaining.startswith(f'"{key}"') or remaining.startswith(f"'{key}"):
                        # 找到顶层关键字，需要在此之前闭合 properties
                        # 在最后一个有效闭合位置之后插入 }
                        if last_valid_close != -1:
                            insert_pos = last_valid_close + 1
                            s = s[:insert_pos] + '}' + s[insert_pos:]
                        return s
        i += 1
    
    return s


def fix_bracket_mismatch(s: str) -> str:
    """
    尝试修复常见的括号不匹配问题
    """
    # 修复模式: 字符串值后直接跟 ,{ 应该是 },{
    s = re.sub(r'(["\'])\s*,\s*(\{)', r'\1},\2', s)
    
    return s


def fix_anyof_structure(s: str) -> str:
    """
    修复 anyOf/oneOf/allOf 结构中的常见问题
    
    问题: {'anyof':[{'type':'string',{'type':'null'},'default':None,'title':'X'}]}
    修复: {'anyOf':[{'type':'string'},{'type':'null'}],'default':None,'title':'X'}
    """
    # 查找所有 anyOf/oneOf/allOf 模式（包括未闭合的数组）
    pattern = r'(["\'])(anyof|oneof|allof)\1\s*:\s*\[([^\]]+?)(\]|$)'
    
    def fix_array(match):
        quote = match.group(1)
        key = match.group(2)
        array_content = match.group(3)
        has_closing = match.group(4) == ']'
        
        # 修复数组内的 ,{ -> },{
        fixed = re.sub(r'(["\'])\s*,\s*(\{)', r'\1},\2', array_content)
        
        # 提取数组外的属性（default, title 等）
        outer_props = []
        remaining = fixed
        
        # 查找所有应该在外层的属性
        for prop in ['default', 'title', 'description']:
            # 匹配 ,'prop': value 或 ,"prop": value
            prop_pattern = rf',\s*(["\']){prop}\1\s*:\s*([^,\]}}]+)'
            matches = list(re.finditer(prop_pattern, remaining))
            if matches:
                # 取最后一个匹配
                last_match = matches[-1]
                outer_props.append(last_match.group(0))
                # 从 remaining 中移除
                remaining = remaining[:last_match.start()] + remaining[last_match.end():]
        
        # 清理 remaining（数组内容）
        # 移除尾随的 } 如果它不属于数组元素
        remaining = remaining.rstrip(',').strip()
        
        # 检查是否有多余的 }
        # 计算 { 和 } 的数量
        open_count = remaining.count('{')
        close_count = remaining.count('}')
        if close_count > open_count:
            # 移除多余的 }
            for _ in range(close_count - open_count):
                # 从末尾移除最后一个 }
                last_brace = remaining.rfind('}')
                if last_brace != -1:
                    remaining = remaining[:last_brace] + remaining[last_brace+1:]
        
        # 重组，保持原始关键字大小写
        result = f'{quote}{key}{quote}:[{remaining}]'
        if outer_props:
            result += ''.join(outer_props)
        
        return result
    
    s = re.sub(pattern, fix_array, s, flags=re.IGNORECASE)
    return s


def fix_enum_values(s: str) -> str:
    """
    修复 enum 数组中缺少引号的值
    
    例如: enum:[complete,partial] -> enum:["complete","partial"]
    """
    # 查找所有 enum: [...] 模式
    pattern = r'(["\']enum["\']\s*:\s*\[)([^\]]+)(\])'
    
    def fix_enum_array(match):
        prefix = match.group(1)
        content = match.group(2)
        suffix = match.group(3)
        
        # 分割数组元素
        items = []
        current = []
        in_string = False
        string_char = None
        
        for c in content:
            if not in_string:
                if c in ('"', "'"):
                    in_string = True
                    string_char = c
                    current.append(c)
                elif c == ',':
                    if current:
                        items.append(''.join(current).strip())
                        current = []
                else:
                    current.append(c)
            else:
                current.append(c)
                if c == string_char:
                    in_string = False
                    string_char = None
        
        if current:
            items.append(''.join(current).strip())
        
        # 修复每个元素：如果不是以引号开头，添加引号
        fixed_items = []
        for item in items:
            item = item.strip()
            if not item:
                continue
            # 如果已经有引号，保持不变
            if item.startswith('"') or item.startswith("'"):
                fixed_items.append(item)
            else:
                # 添加引号
                fixed_items.append(f'"{item}"')
        
        return prefix + ','.join(fixed_items) + suffix
    
    s = re.sub(pattern, fix_enum_array, s, flags=re.IGNORECASE)
    return s


def fix_json_string(s: str) -> str:
    """
    修复常见的JSON格式问题
    
    处理的问题：
    1. 中文标点
    2. 未加引号的key
    3. 单引号转双引号
    4. Python None/True/False
    5. 尾随逗号
    6. JSON Schema 关键字大小写
    7. 括号不匹配
    8. anyOf/oneOf/allOf 结构错误
    9. enum 值缺少引号
    """
    if not s:
        return s
    
    # 步骤1: 替换中文标点
    s = fix_chinese_punctuation(s)
    
    # 步骤1.5: 修复 enum 值缺少引号
    s = fix_enum_values(s)
    
    # 步骤2: 修复 properties 闭合问题
    s = fix_properties_closure(s)
    
    # 步骤3: 尝试修复括号不匹配
    s = fix_bracket_mismatch(s)
    
    # 步骤4: 修复 anyOf/oneOf/allOf 结构
    s = fix_anyof_structure(s)
    
    # 步骤3: 修复未加引号的key（在 { 或 , 后面的标识符后跟着 :）
    s = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', s)
    
    # 步骤4: 将单引号转换为双引号（小心处理字符串内容）
    result = []
    in_string = False
    string_char = None
    i = 0
    while i < len(s):
        c = s[i]
        
        if not in_string:
            if c == '"':
                in_string = True
                string_char = '"'
                result.append(c)
            elif c == "'":
                in_string = True
                string_char = "'"
                result.append('"')  # 单引号转双引号
            else:
                result.append(c)
        else:
            if c == '\\' and i + 1 < len(s):
                # 转义字符
                result.append(c)
                result.append(s[i + 1])
                i += 2
                continue
            elif c == string_char:
                in_string = False
                string_char = None
                result.append('"')  # 统一转为双引号
            else:
                result.append(c)
        i += 1
    
    s = ''.join(result)
    
    # 步骤5: 处理 Python 的 None, True, False
    s = re.sub(r'\bNone\b', 'null', s)
    s = re.sub(r'\bTrue\b', 'true', s)
    s = re.sub(r'\bFalse\b', 'false', s)
    
    # 步骤6: 移除尾随逗号 (,] 或 ,})
    s = re.sub(r',\s*([\]\}])', r'\1', s)
    
    # 步骤7: 不修复 JSON Schema 关键字大小写，保持原样
    # s = re.sub(r'"anyof":', '"anyOf":', s, flags=re.IGNORECASE)
    # s = re.sub(r'"oneof":', '"oneOf":', s, flags=re.IGNORECASE)
    # s = re.sub(r'"allof":', '"allOf":', s, flags=re.IGNORECASE)
    
    return s


def normalize_quotes(text: str) -> str:
    """
    将文本中的单引号统一转换为双引号
    小心处理转义字符，避免破坏字符串内容
    
    Args:
        text: 要标准化的文本
        
    Returns:
        引号标准化后的文本
    """
    result = []
    in_string = False
    string_char = None
    i = 0
    
    while i < len(text):
        c = text[i]
        
        if not in_string:
            if c == '"':
                in_string = True
                string_char = '"'
                result.append(c)
            elif c == "'":
                in_string = True
                string_char = "'"
                result.append('"')  # 单引号转双引号
            else:
                result.append(c)
            i += 1
        else:
            # 在字符串内部
            if c == '\\' and i + 1 < len(text):
                # 转义字符：保留转义序列
                result.append(c)
                result.append(text[i + 1])
                i += 2
            elif c == string_char:
                # 字符串结束
                in_string = False
                string_char = None
                result.append('"')  # 统一转为双引号
                i += 1
            else:
                result.append(c)
                i += 1
    
    return ''.join(result)


def extract_balanced_braces(text: str, start_pos: int = 0, normalize: bool = True) -> Optional[str]:
    """
    提取从 start_pos 开始的平衡花括号内容
    
    Args:
        text: 要搜索的文本
        start_pos: 开始位置
        normalize: 是否先标准化引号（默认True）
        
    Returns:
        平衡的花括号内容（包含外层花括号），如果没有找到则返回 None
    """
    # 先将单引号统一转换为双引号，避免混合引号导致的解析问题
    if normalize:
        text = normalize_quotes(text)
    
    # 从start_pos开始查找第一个{
    brace_start = text.find('{', start_pos)
    if brace_start == -1:
        return None
    
    # 跟踪花括号的嵌套深度
    count = 0
    in_string = False
    escape_next = False
    
    for i in range(brace_start, len(text)):
        c = text[i]
        
        if escape_next:
            escape_next = False
            continue
        
        if c == '\\':
            escape_next = True
            continue
        
        # 跟踪字符串状态（现在只需要处理双引号）
        if c == '"':
            in_string = not in_string
        elif not in_string:
            if c == '{':
                count += 1
            elif c == '}':
                count -= 1
                if count == 0:
                    # 找到匹配的闭合括号
                    return text[brace_start:i+1]
    
    # 没有找到匹配的闭合括号
    return None


def parse_schema_string(schema_str: str) -> Optional[Dict]:
    """
    尝试多种方法解析 schema 字符串
    
    Args:
        schema_str: 原始 schema 字符串
        
    Returns:
        解析后的字典，失败返回 None
    """
    if not schema_str or not schema_str.strip():
        return None
    
    # 预处理：移除首尾空白
    schema_str = schema_str.strip()
    
    # 策略1: 先标准化引号，再使用fix_json_string + json.loads
    try:
        normalized = normalize_quotes(schema_str)
        fixed = fix_json_string(normalized)
        result = json.loads(fixed)
        # 检查是否为空字典（可能是解析错误）
        if result and isinstance(result, dict):
            return result
    except Exception as e:
        pass
    
    # 策略2: ast.literal_eval（处理 Python dict 字面量）
    try:
        result = ast.literal_eval(schema_str)
        if result and isinstance(result, dict):
            return result
    except Exception as e:
        pass
    
    # 策略3: 简单替换后尝试
    try:
        simple = schema_str.replace('None', 'null').replace('True', 'true').replace('False', 'false')
        simple = normalize_quotes(simple)
        result = json.loads(simple)
        if result and isinstance(result, dict):
            return result
    except Exception as e:
        pass
    
    # 策略4: 使用fix_json_string但不标准化引号（处理已经是双引号的情况）
    try:
        fixed = fix_json_string(schema_str)
        result = json.loads(fixed)
        if result and isinstance(result, dict):
            return result
    except Exception as e:
        pass
    
    return None


def check_field_loss(original_text: str, parsed_schema: Dict, tool_name: str, schema_type: str) -> List[str]:
    """
    检查原始文本中的关键字段是否在解析结果中丢失
    
    策略：
    1. 提取原始文本中出现的所有字段名（在引号中的）
    2. 检查这些字段名是否在解析后的schema中存在
    3. 只报告丢失的字段
    
    Args:
        original_text: 原始文本片段
        parsed_schema: 解析后的schema
        tool_name: 工具名
        schema_type: 'Parameters' 或 'Output'
        
    Returns:
        丢失字段的警告列表
    """
    warnings = []
    
    if not original_text or not parsed_schema:
        return warnings
    
    # 提取原始文本中的所有字段名（在引号中的）
    # 匹配模式: "field_name" 或 'field_name' 后面跟着 :
    field_pattern = r'["\']([a-zA-Z_][a-zA-Z0-9_]*)["\']\s*:'
    original_fields = set(re.findall(field_pattern, original_text))
    
    # 递归提取解析后的schema中的所有字段名
    def extract_all_keys(obj, keys=None):
        if keys is None:
            keys = set()
        if isinstance(obj, dict):
            keys.update(obj.keys())
            for value in obj.values():
                extract_all_keys(value, keys)
        elif isinstance(obj, list):
            for item in obj:
                extract_all_keys(item, keys)
        return keys
    
    parsed_fields = extract_all_keys(parsed_schema)
    
    # 找出丢失的字段
    lost_fields = original_fields - parsed_fields
    
    # 过滤掉一些常见的非关键字段（可能是误报）
    # 例如：如果原文本中有 "type":"string" 但解析后变成 type: "string"，
    # 那么 "string" 也会被匹配为字段名，但实际上它是值
    common_values = {'string', 'integer', 'number', 'boolean', 'array', 'object', 'null',
                     'low', 'medium', 'high', 'complete', 'partial', 'true', 'false'}
    lost_fields = lost_fields - common_values
    
    if lost_fields:
        for field in sorted(lost_fields):
            warnings.append(f"工具 {tool_name} - {schema_type}.{field} - 字段在解析后丢失")
    
    return warnings


def validate_schema(schema: Dict, schema_type: str, tool_name: str) -> List[str]:
    """
    验证 schema 的完整性和有效性
    注意：此函数已废弃，改用 compare_schemas 进行对比验证
    
    Args:
        schema: 要验证的 schema 字典
        schema_type: 'Parameters' 或 'Output'
        tool_name: 工具名
        
    Returns:
        警告信息列表
    """
    # 此函数保留但不再使用，避免破坏现有代码
    return []


def normalize_tool_txt(input_path: str, output_path: str, verbose: bool = True, debug: bool = False):
    """
    标准化 tool.txt 文件
    
    Args:
        input_path: 原始 tool.txt 路径
        output_path: 输出的标准化文件路径
        verbose: 是否打印详细信息
        debug: 是否输出调试信息（包括原始文本片段）
        
    Returns:
        (成功标志, 解析错误列表)
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按工具分割（每个工具以数字+点开头）
    tool_blocks = re.split(r'\n(?=\d+\.\s+Name:)', content.strip())
    
    normalized_tools = []
    parse_errors = []
    parse_warnings = []
    validation_warnings = []
    debug_info = []  # 存储调试信息
    
    for block in tool_blocks:
        if not block.strip():
            continue
        
        # 提取工具编号
        num_match = re.search(r'^(\d+)\.\s+Name:', block)
        tool_num = num_match.group(1) if num_match else '?'
        
        # 提取工具名
        name_match = re.search(r'Name:\s*(\w+)', block)
        if not name_match:
            parse_errors.append(f"工具 {tool_num} - 无法提取工具名")
            continue
        tool_name = name_match.group(1)
        
        # 提取描述（支持多行）
        desc_match = re.search(r'Description:\s*(.+?)(?=\nParameters:|\nOutput:|$)', block, re.DOTALL)
        description = desc_match.group(1).strip() if desc_match else ''
        
        # 清理描述中的多余空白
        description = ' '.join(description.split())
        
        # 提取并解析 Parameters
        params_schema = None
        params_pos = block.find('Parameters:')
        params_raw_text = ''  # 记录原始文本
        if params_pos != -1:
            # 提取Parameters后的原始文本（用于调试和对比）
            params_line_end = block.find('\n', params_pos)
            if params_line_end != -1:
                params_raw_text = block[params_pos:params_line_end].strip()
            else:
                params_raw_text = block[params_pos:].strip()
            
            # 提取标准化后的文本
            params_str = extract_balanced_braces(block, params_pos, normalize=True)
            
            if params_str:
                # 解析标准化后的文本
                params_schema = parse_schema_string(params_str)
                
                if params_schema is None:
                    parse_errors.append(f"工具 {tool_num} ({tool_name}) - Parameters 解析失败")
                    if debug:
                        debug_info.append(f"工具 {tool_num} ({tool_name}) - Parameters 原始文本: {params_raw_text}")
                        debug_info.append(f"  提取的内容: {params_str[:200]}...")
            else:
                parse_warnings.append(f"工具 {tool_num} ({tool_name}) - Parameters 未找到花括号")
                if debug:
                    debug_info.append(f"工具 {tool_num} ({tool_name}) - Parameters 原始文本: {params_raw_text}")
        
        # 提取并解析 Output
        output_schema = None
        output_pos = block.find('Output:')
        output_raw_text = ''  # 记录原始文本
        if output_pos != -1:
            # 提取Output后的原始文本（用于调试和对比）
            output_line_end = block.find('\n', output_pos)
            if output_line_end != -1:
                output_raw_text = block[output_pos:output_line_end].strip()
            else:
                output_raw_text = block[output_pos:].strip()
            
            # 提取标准化后的文本
            output_str = extract_balanced_braces(block, output_pos, normalize=True)
            
            if output_str:
                # 解析标准化后的文本
                output_schema = parse_schema_string(output_str)
                
                if output_schema is None:
                    parse_errors.append(f"工具 {tool_num} ({tool_name}) - Output 解析失败")
                    if debug:
                        debug_info.append(f"工具 {tool_num} ({tool_name}) - Output 原始文本: {output_raw_text}")
                        debug_info.append(f"  提取的内容: {output_str[:200]}...")
                elif not output_schema or output_schema == {}:
                    # 检测静默失败：解析成功但结果为空字典
                    parse_warnings.append(f"工具 {tool_num} ({tool_name}) - Output 解析为空字典")
                    if debug:
                        debug_info.append(f"工具 {tool_num} ({tool_name}) - Output 原始文本: {output_raw_text}")
                        debug_info.append(f"  提取的内容: {output_str[:200]}...")
            else:
                parse_warnings.append(f"工具 {tool_num} ({tool_name}) - Output 未找到花括号")
                if debug:
                    debug_info.append(f"工具 {tool_num} ({tool_name}) - Output 原始文本: {output_raw_text}")
                    # 输出Output:后面的内容供诊断
                    snippet = output_raw_text[7:].strip()[:100]  # 跳过'Output:'
                    debug_info.append(f"  Output: 后的内容: '{snippet}'")
        
        # 构建标准化的工具定义
        normalized_block = f"{tool_num}. Name: {tool_name}\n"
        normalized_block += f"Description: {description}\n"
        
        if params_schema:
            # 使用紧凑格式，但保持可读性
            normalized_block += f"Parameters: {json.dumps(params_schema, ensure_ascii=False, separators=(',', ': '))}\n"
        else:
            normalized_block += "Parameters: {}\n"
        
        if output_schema:
            normalized_block += f"Output: {json.dumps(output_schema, ensure_ascii=False, separators=(',', ': '))}\n"
        else:
            normalized_block += "Output: {}\n"
        
        normalized_tools.append(normalized_block)
        
        # 检查原始文本中的字段是否在解析后丢失
        if params_schema and params_raw_text:
            param_warnings = check_field_loss(params_raw_text, params_schema, tool_name, 'Parameters')
            validation_warnings.extend(param_warnings)
        
        if output_schema and output_raw_text:
            output_warnings = check_field_loss(output_raw_text, output_schema, tool_name, 'Output')
            validation_warnings.extend(output_warnings)
    
    # 写入标准化文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(normalized_tools))
    
    # 打印报告
    if verbose:
        print("=" * 60)
        print("标准化完成！")
        print("=" * 60)
        print(f"输入文件: {input_path}")
        print(f"输出文件: {output_path}")
        print(f"处理工具数: {len(normalized_tools)}")
        
        if parse_warnings:
            print(f"\n⚠ 警告 ({len(parse_warnings)} 个):")
            for warn in parse_warnings:
                print(f"  {warn}")
        
        if validation_warnings:
            print(f"\n⚠ 质量检查警告 ({len(validation_warnings)} 个):")
            for warn in validation_warnings:
                print(f"  {warn}")
        
        if parse_errors:
            print(f"\n✗ 解析错误 ({len(parse_errors)} 个):")
            for err in parse_errors:
                print(f"  {err}")
        else:
            print("\n✓ 所有工具 schema 解析成功！")
        
        if debug and debug_info:
            print(f"\n🔍 调试信息 ({len(debug_info)} 条):")
            for info in debug_info:
                print(f"  {info}")
        
        print("=" * 60)
        
        if len(parse_errors) == 0:
            print(f"\n建议：")
            print(f"  1. 检查 {output_path} 确认格式正确")
            print(f"  2. 备份原文件: cp {input_path} {input_path}.bak")
            print(f"  3. 替换原文件: cp {output_path} {input_path}")
            print(f"  或在其他脚本中直接使用 {output_path}")
    
    return len(parse_errors) == 0, parse_errors


if __name__ == "__main__":
    input_file = "/Users/liaoying/Desktop/研一/llm/data_eval/precision_index/tool.txt"
    output_file = "/Users/liaoying/Desktop/研一/llm/data_eval/precision_index/tool_normalized.txt"
    
    # 设置 debug=True 可以看到详细的调试信息
    success, errors = normalize_tool_txt(input_file, output_file, verbose=True, debug=True)
    
    if not success:
        print("\n需要手动修复以下问题后重新运行:")
        for err in errors:
            print(f"  - {err}")
        exit(1)
    else:
        print("\n✓ 标准化成功！")
        exit(0)
