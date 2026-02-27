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
        
        # 重组
        result = f'{quote}{key.lower()}{quote}:[{remaining}]'
        if outer_props:
            result += ''.join(outer_props)
        
        # 修正关键字大小写
        if key.lower() == 'anyof':
            result = result.replace(f'{quote}anyof{quote}', f'{quote}anyOf{quote}')
        elif key.lower() == 'oneof':
            result = result.replace(f'{quote}oneof{quote}', f'{quote}oneOf{quote}')
        elif key.lower() == 'allof':
            result = result.replace(f'{quote}allof{quote}', f'{quote}allOf{quote}')
        
        return result
    
    s = re.sub(pattern, fix_array, s, flags=re.IGNORECASE)
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
    """
    if not s:
        return s
    
    # 步骤1: 替换中文标点
    s = fix_chinese_punctuation(s)
    
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
    
    # 步骤7: 修复常见的 JSON Schema 关键字大小写
    s = re.sub(r'"anyof":', '"anyOf":', s, flags=re.IGNORECASE)
    s = re.sub(r'"oneof":', '"oneOf":', s, flags=re.IGNORECASE)
    s = re.sub(r'"allof":', '"allOf":', s, flags=re.IGNORECASE)
    
    return s


def extract_balanced_braces(text: str, start_pos: int = 0) -> Optional[str]:
    """
    提取从 start_pos 开始的平衡花括号内容
    
    Args:
        text: 要搜索的文本
        start_pos: 开始位置
        
    Returns:
        平衡的花括号内容（包含外层花括号），如果没有找到则返回 None
    """
    brace_start = text.find('{', start_pos)
    if brace_start == -1:
        return None
    
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


def parse_schema_string(schema_str: str) -> Optional[Dict]:
    """
    尝试多种方法解析 schema 字符串
    
    Args:
        schema_str: 原始 schema 字符串
        
    Returns:
        解析后的字典，失败返回 None
    """
    # 方法1: fix_json_string + json.loads
    try:
        fixed = fix_json_string(schema_str)
        return json.loads(fixed)
    except Exception as e:
        pass
    
    # 方法2: ast.literal_eval（处理 Python dict 字面量）
    try:
        return ast.literal_eval(schema_str)
    except Exception as e:
        pass
    
    # 方法3: 简单替换后尝试
    try:
        simple = schema_str.replace('None', 'null').replace("'", '"')
        return json.loads(simple)
    except Exception as e:
        pass
    
    return None


def validate_schema_quality(schema: Dict, schema_type: str) -> List[str]:
    """
    验证 schema 质量，返回警告列表
    
    Args:
        schema: 要验证的 schema
        schema_type: schema 类型（'Parameters' 或 'Output'）
        
    Returns:
        警告信息列表
    """
    warnings = []
    
    if not schema:
        warnings.append(f"{schema_type} 为空")
        return warnings
    
    # 检查 Parameters schema
    if schema_type == 'Parameters':
        # 检查是否有 properties
        if 'properties' not in schema or not schema['properties']:
            warnings.append(f"{schema_type} 缺少 properties 或 properties 为空")
        else:
            # 检查每个属性是否有 type 或 title
            for prop_name, prop_schema in schema['properties'].items():
                if not isinstance(prop_schema, dict):
                    warnings.append(f"{schema_type}.properties.{prop_name} 不是对象")
                    continue
                
                if 'type' not in prop_schema and 'anyOf' not in prop_schema and 'oneOf' not in prop_schema:
                    warnings.append(f"{schema_type}.properties.{prop_name} 缺少 type 定义")
                
                if 'title' not in prop_schema:
                    warnings.append(f"{schema_type}.properties.{prop_name} 缺少 title")
        
        # 检查是否有 type
        if 'type' not in schema:
            warnings.append(f"{schema_type} 缺少顶层 type 定义")
        
        # 检查 required 字段
        if 'required' in schema:
            if not isinstance(schema['required'], list):
                warnings.append(f"{schema_type}.required 应该是数组")
            elif not schema['required']:
                warnings.append(f"{schema_type}.required 为空数组")
    
    # 检查 Output schema
    elif schema_type == 'Output':
        # 检查是否有 properties 或 type
        if 'properties' not in schema and 'type' not in schema:
            warnings.append(f"{schema_type} 缺少 properties 或 type 定义")
        
        # 如果有 properties，检查每个属性
        if 'properties' in schema:
            if not schema['properties']:
                warnings.append(f"{schema_type}.properties 为空")
            else:
                for prop_name, prop_schema in schema['properties'].items():
                    if not isinstance(prop_schema, dict):
                        warnings.append(f"{schema_type}.properties.{prop_name} 不是对象")
                        continue
                    
                    if 'type' not in prop_schema and 'anyOf' not in prop_schema:
                        warnings.append(f"{schema_type}.properties.{prop_name} 缺少 type 定义")
    
    return warnings


def normalize_tool_txt(input_path: str, output_path: str, verbose: bool = True):
    """
    标准化 tool.txt 文件
    
    Args:
        input_path: 原始 tool.txt 路径
        output_path: 输出的标准化文件路径
        verbose: 是否打印详细信息
        
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
    quality_warnings = []
    
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
        if params_pos != -1:
            params_str = extract_balanced_braces(block, params_pos)
            if params_str:
                params_schema = parse_schema_string(params_str)
                if params_schema is None:
                    parse_errors.append(f"工具 {tool_num} ({tool_name}) - Parameters 解析失败")
                    if verbose:
                        print(f"  原始 Parameters: {params_str[:100]}...")
            else:
                parse_warnings.append(f"工具 {tool_num} ({tool_name}) - Parameters 未找到花括号")
        
        # 提取并解析 Output
        output_schema = None
        output_pos = block.find('Output:')
        if output_pos != -1:
            output_str = extract_balanced_braces(block, output_pos)
            if output_str:
                output_schema = parse_schema_string(output_str)
                if output_schema is None:
                    parse_errors.append(f"工具 {tool_num} ({tool_name}) - Output 解析失败")
                    if verbose:
                        print(f"  原始 Output: {output_str[:100]}...")
            else:
                parse_warnings.append(f"工具 {tool_num} ({tool_name}) - Output 未找到花括号")
        
        # === 质量检测 ===
        # 检查描述
        if not description or description in ['xxx', 'XXX', 'TODO', '']:
            quality_warnings.append(f"工具 {tool_num} ({tool_name}) - Description 为空或占位符")
        
        # 检查 Parameters schema 质量
        if params_schema:
            param_warnings = validate_schema_quality(params_schema, 'Parameters')
            for warn in param_warnings:
                quality_warnings.append(f"工具 {tool_num} ({tool_name}) - {warn}")
        else:
            quality_warnings.append(f"工具 {tool_num} ({tool_name}) - Parameters 为空")
        
        # 检查 Output schema 质量
        if output_schema:
            output_warnings = validate_schema_quality(output_schema, 'Output')
            for warn in output_warnings:
                quality_warnings.append(f"工具 {tool_num} ({tool_name}) - {warn}")
        else:
            quality_warnings.append(f"工具 {tool_num} ({tool_name}) - Output 为空")
        
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
            print(f"\n⚠ 解析警告 ({len(parse_warnings)} 个):")
            for warn in parse_warnings:
                print(f"  • {warn}")
        
        if parse_errors:
            print(f"\n✗ 解析错误 ({len(parse_errors)} 个):")
            for err in parse_errors:
                print(f"  • {err}")
        else:
            print("\n✓ 所有工具 schema 解析成功！")
        
        # 打印质量检测结果
        if quality_warnings:
            print(f"\n⚠ 质量检测警告 ({len(quality_warnings)} 个):")
            print("  以下问题不影响 JSON 解析，但可能影响 schema 的完整性和可用性：")
            
            # 按工具分组显示
            warnings_by_tool = {}
            for warn in quality_warnings:
                # 提取工具编号
                match = re.match(r'工具 (\d+) \(([^)]+)\) - (.+)', warn)
                if match:
                    tool_num = match.group(1)
                    tool_name = match.group(2)
                    warning_msg = match.group(3)
                    key = f"工具 {tool_num} ({tool_name})"
                    if key not in warnings_by_tool:
                        warnings_by_tool[key] = []
                    warnings_by_tool[key].append(warning_msg)
            
            for tool_key, warnings in warnings_by_tool.items():
                print(f"\n  {tool_key}:")
                for w in warnings:
                    print(f"    - {w}")
        else:
            print("\n✓ 所有工具 schema 质量检测通过！")
        
        print("=" * 60)
        
        if len(parse_errors) == 0:
            print(f"\n建议：")
            print(f"  1. 检查 {output_path} 确认格式正确")
            print(f"  2. 备份原文件: cp {input_path} {input_path}.bak")
            print(f"  3. 替换原文件: cp {output_path} {input_path}")
            print(f"  或在其他脚本中直接使用 {output_path}")
    
    return len(parse_errors) == 0, parse_errors


if __name__ == "__main__":
    input_file = "/mnt/data/kw/ly/precision_index/tool.txt"
    output_file = "/mnt/data/kw/ly/precision_index/tool_normalized.txt"
    
    success, errors = normalize_tool_txt(input_file, output_file, verbose=True)
    
    if not success:
        print("\n需要手动修复以下问题后重新运行:")
        for err in errors:
            print(f"  - {err}")
        exit(1)
    else:
        print("\n✓ 标准化成功！")
        exit(0)
