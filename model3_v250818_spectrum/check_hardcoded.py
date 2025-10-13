#!/usr/bin/env python3
"""
配置验证脚本
检查项目中是否还有硬编码的365或其他配置参数
"""

import os
import re
from pathlib import Path

def find_hardcoded_values(directory):
    """查找硬编码的值"""
    patterns = {
        '365': r'\b365\b',  # 365作为独立数字
        'batch_size=64': r'batch_size\s*=\s*64',
        'num_epochs=1000': r'num_epochs\s*=\s*1000',
        'hidden_size=128': r'hidden_size\s*=\s*128',
        'num_layers=2': r'num_layers\s*=\s*2',
        'lr=0.0001': r'lr\s*=\s*0\.0001',
        'patience=30': r'patience\s*=\s*30',
    }
    
    results = {}
    
    # 要检查的文件类型
    file_extensions = ['.py', '.ipynb']
    
    for root, dirs, files in os.walk(directory):
        # 跳过配置文件本身和__pycache__
        if '__pycache__' in root or 'config.py' in files:
            continue
            
        for file in files:
            if any(file.endswith(ext) for ext in file_extensions):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, directory)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    for pattern_name, pattern in patterns.items():
                        matches = re.finditer(pattern, content, re.MULTILINE)
                        for match in matches:
                            # 计算行号
                            line_num = content[:match.start()].count('\n') + 1
                            line_content = content.split('\n')[line_num - 1].strip()
                            
                            if rel_path not in results:
                                results[rel_path] = []
                            
                            results[rel_path].append({
                                'pattern': pattern_name,
                                'line': line_num,
                                'content': line_content,
                                'match': match.group()
                            })
                            
                except Exception as e:
                    print(f"无法读取文件 {file_path}: {e}")
    
    return results

def main():
    """主函数"""
    print("=== 硬编码检查报告 ===\n")
    
    # 检查当前目录
    current_dir = Path(__file__).parent
    results = find_hardcoded_values(current_dir)
    
    if not results:
        print("✅ 太棒了！没有发现硬编码的配置参数。")
        return
    
    print("❌ 发现以下硬编码的配置参数:\n")
    
    for file_path, issues in results.items():
        print(f"📁 文件: {file_path}")
        print("-" * 50)
        
        for issue in issues:
            print(f"  🔍 第{issue['line']}行: {issue['pattern']}")
            print(f"     内容: {issue['content']}")
            print(f"     匹配: {issue['match']}")
            print()
        
        print()
    
    print("建议:")
    print("1. 将这些硬编码值替换为 config.py 中的配置常量")
    print("2. 确保所有函数都使用配置参数作为默认值")
    print("3. 在函数调用时显式传递配置参数")

if __name__ == "__main__":
    main()