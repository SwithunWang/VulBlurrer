#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
from collections import defaultdict

def parse_filename(filename):
    """
    解析文件名，提取模型名称、轮次和进程信息
    """
    # 移除文件扩展名
    base_name = os.path.splitext(filename)[0]
    
    # 提取模型名称
    model_pattern = r'(codebert|graphcodebert|codeT5)'
    model_match = re.search(model_pattern, base_name)
    model = model_match.group(1) if model_match else None
    
    # 提取轮次信息
    take_pattern = r'take_(\d+)'
    take_match = re.search(take_pattern, base_name)
    take = take_match.group(1) if take_match else None
    
    # 提取进程信息
    process_pattern = r'process_(\d+)'
    process_match = re.search(process_pattern, base_name)
    process = process_match.group(1) if process_match else None
    
    return {
        'model': model,
        'take': take,
        'process': process,
        'filename': filename
    }

def consolidate_files(results_dir):
    """
    整理指定目录中的文件，将同一模型相同轮次不同进程的文件合并
    """
    # 获取所有.jsonl文件
    files = [f for f in os.listdir(results_dir) if f.endswith('.jsonl')]
    
    # 按模型和轮次分组
    groups = defaultdict(lambda: defaultdict(list))
    
    # 解析每个文件名并分组
    for file in files:
        info = parse_filename(file)
        if info['model'] and info['take']:
            key = f"{info['model']}_take_{info['take']}"
            groups[info['model']][info['take']].append({
                'filename': file,
                'process': info['process']
            })
    
    # 为每个分组创建合并文件
    for model, takes in groups.items():
        for take, files in takes.items():
            # 输出文件名
            output_filename = f"consolidated_{model}_take_{take}.jsonl"
            output_path = os.path.join(results_dir, output_filename)
            
            print(f"正在合并 {model} 模型第 {take} 轮的文件:")
            consolidated_data = []
            
            # 读取每个文件的内容
            for file_info in files:
                file_path = os.path.join(results_dir, file_info['filename'])
                print(f"  - 处理文件: {file_info['filename']}")
                
                # 读取文件中的每一行
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:  # 跳过空行
                            try:
                                data = json.loads(line)
                                consolidated_data.append(data)
                            except json.JSONDecodeError:
                                print(f"    警告: 无法解析行: {line}")
            
            # 写入合并后的数据
            with open(output_path, 'w', encoding='utf-8') as f:
                for data in consolidated_data:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
            
            print(f"  合并完成，共 {len(consolidated_data)} 条记录，保存至: {output_filename}")
            print()

def main():
    results_dir = ""
    
    if not os.path.exists(results_dir):
        print(f"错误: 目录 {results_dir} 不存在")
        return
    
    print("开始整理攻击结果文件...")
    consolidate_files(results_dir)
    print("所有文件整理完成!")

if __name__ == "__main__":
    main()