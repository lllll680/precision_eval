#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量重命名JSON文件脚本

功能：
- 遍历指定的多个文件夹
- 跳过 question_info.json 和 batch_summary.json
- 将其他JSON文件重命名为 run_001.json, run_002.json, ..., run_020.json
- 每个文件夹独立编号（都从 run_001.json 开始）

使用方法：
    python rename_json_files.py
"""

import os
import json
from pathlib import Path
from typing import List


def rename_json_files_in_folder(folder_path: str, dry_run: bool = True):
    """
    重命名指定文件夹中的JSON文件（两阶段重命名策略）
    
    策略说明：
    1. 第一阶段：将所有文件重命名为临时名称（temp_001.json, temp_002.json, ...）
    2. 第二阶段：将临时文件重命名为最终名称（run_001.json, run_002.json, ...）
    
    这样可以避免文件名冲突导致的编号不连续问题
    
    Args:
        folder_path: 文件夹路径
        dry_run: 是否为试运行模式（True=只显示不执行，False=实际重命名）
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"❌ 文件夹不存在: {folder_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"处理文件夹: {folder_path}")
    print(f"{'='*60}")
    
    # 获取所有JSON文件（排除特殊文件）
    json_files = []
    for json_file in sorted(folder.glob("*.json")):
        if json_file.name in ['question_info.json', 'batch_summary.json']:
            print(f"⏭️  跳过特殊文件: {json_file.name}")
            continue
        json_files.append(json_file)
    
    if not json_files:
        print("⚠️  没有找到需要重命名的JSON文件")
        return
    
    print(f"\n找到 {len(json_files)} 个JSON文件需要重命名")
    print(f"使用两阶段重命名策略避免文件名冲突\n")
    
    # 第一阶段：重命名为临时文件名
    print("[阶段1] 重命名为临时文件名...")
    temp_files = []
    stage1_count = 0
    
    for idx, old_file in enumerate(json_files, start=1):
        temp_name = f"temp_{idx:03d}.json"
        temp_path = folder / temp_name
        
        if dry_run:
            print(f"  🔍 [试运行] {old_file.name} -> {temp_name}")
            temp_files.append((temp_path, idx))
        else:
            try:
                old_file.rename(temp_path)
                temp_files.append((temp_path, idx))
                stage1_count += 1
            except Exception as e:
                print(f"  ❌ 重命名失败: {old_file.name} -> {temp_name}, 错误: {e}")
    
    if not dry_run:
        print(f"  ✅ 阶段1完成，成功重命名 {stage1_count} 个文件\n")
    else:
        print(f"  [试运行] 阶段1将重命名 {len(temp_files)} 个文件\n")
    
    # 第二阶段：重命名为最终文件名
    print("[阶段2] 重命名为最终文件名...")
    stage2_count = 0
    
    for temp_path, idx in temp_files:
        final_name = f"run_{idx:03d}.json"
        final_path = folder / final_name
        
        if dry_run:
            print(f"  🔍 [试运行] {temp_path.name} -> {final_name}")
        else:
            try:
                temp_path.rename(final_path)
                print(f"  ✅ {temp_path.name} -> {final_name}")
                stage2_count += 1
            except Exception as e:
                print(f"  ❌ 重命名失败: {temp_path.name} -> {final_name}, 错误: {e}")
    
    if dry_run:
        print(f"\n[试运行模式] 将重命名 {len(json_files)} 个文件")
    else:
        print(f"\n✅ 成功重命名 {stage2_count} 个文件")


def batch_rename_folders(folder_paths: List[str], dry_run: bool = True):
    """
    批量处理多个文件夹
    
    Args:
        folder_paths: 文件夹路径列表
        dry_run: 是否为试运行模式
    """
    print("="*60)
    print("批量重命名JSON文件")
    print("="*60)
    print(f"模式: {'试运行（不会实际修改文件）' if dry_run else '实际执行（会修改文件）'}")
    print(f"待处理文件夹数量: {len(folder_paths)}")
    
    for folder_path in folder_paths:
        rename_json_files_in_folder(folder_path, dry_run)
    
    print("\n" + "="*60)
    print("批量重命名完成")
    print("="*60)
    
    if dry_run:
        print("\n⚠️  这是试运行模式，文件未被实际修改")
        print("如需实际执行，请将 dry_run=False")


if __name__ == "__main__":
    # 配置要处理的文件夹列表
    folders = [
        "/Users/liaoying/Desktop/研一/llm/data_eval/precision_index/data1",
        #"/Users/liaoying/Desktop/研一/llm/data_eval/precision_index/data2",
        # 添加更多文件夹...
    ]
    
    # 试运行模式（只显示不执行）
    print("\n🔍 第一步：试运行模式，查看将要进行的重命名操作")
    batch_rename_folders(folders, dry_run=True)
    
    # 确认后执行实际重命名
    print("\n" + "="*60)
    user_input = input("是否执行实际重命名？(yes/no): ").strip().lower()
    
    if user_input in ['yes', 'y']:
        print("\n✅ 开始实际重命名...")
        batch_rename_folders(folders, dry_run=False)
    else:
        print("\n❌ 已取消操作")
