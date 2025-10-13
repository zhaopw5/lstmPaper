#!/usr/bin/env python3
"""
完整配置修复脚本
系统性地修复所有硬编码问题
"""

import re
from pathlib import Path

def fix_lstm_cosmic_ray():
    """修复 lstm_cosmic_ray.py 中的硬编码"""
    file_path = Path("lstm_cosmic_ray.py")
    
    if not file_path.exists():
        print(f"文件 {file_path} 不存在")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换模式
    replacements = [
        # 确保所有的365都被替换
        (r'sequence_length\s*=\s*365', 'sequence_length=SEQUENCE_LENGTH'),
        (r'(\d+)\s*天序列', f'{{sequence_length}} 天序列'),  # 注释中的365天
        # 其他硬编码参数
        (r'batch_size\s*=\s*64', 'batch_size=BATCH_SIZE'),
        (r'num_epochs\s*=\s*1000', 'num_epochs=MAX_EPOCHS'),
        (r'hidden_size\s*=\s*128', 'hidden_size=HIDDEN_SIZE'),
        (r'num_layers\s*=\s*2', 'num_layers=NUM_LAYERS'),
        (r'dropout\s*=\s*0\.05', 'dropout=DROPOUT_RATE'),
    ]
    
    modified = False
    for pattern, replacement in replacements:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            modified = True
            print(f"✅ 替换了模式: {pattern}")
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ 已更新 {file_path}")
    else:
        print(f"ℹ️ {file_path} 无需修改")

def fix_data_processor():
    """修复 data_processor.py 中的硬编码"""
    file_path = Path("data_processor.py")
    
    if not file_path.exists():
        print(f"文件 {file_path} 不存在")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换模式
    replacements = [
        # 确保所有的365都被替换
        (r'sequence_length\s*=\s*365', 'sequence_length=SEQUENCE_LENGTH'),
        # 数据划分比例
        (r'train_ratio\s*=\s*0\.70', 'train_ratio=TRAIN_RATIO'),
        (r'val_ratio\s*=\s*0\.15', 'val_ratio=VAL_RATIO'),
        (r'test_ratio\s*=\s*0\.15', 'test_ratio=TEST_RATIO'),
        # 文件路径
        (r'"\.\.\/data\/outputs\/cycle_analysis\/solar_physics_data_1985_2025_v250820\.csv"', 'SOLAR_DATA_PATH'),
        (r'"\.\.\/data\/raw_data\/ams\/helium\.csv"', 'COSMIC_DATA_PATH'),
    ]
    
    modified = False
    for pattern, replacement in replacements:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            modified = True
            print(f"✅ 替换了模式: {pattern}")
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ 已更新 {file_path}")
    else:
        print(f"ℹ️ {file_path} 无需修改")

def verify_config_usage():
    """验证配置是否正确使用"""
    files_to_check = ["lstm_cosmic_ray.py", "data_processor.py"]
    
    print("\n=== 验证配置使用情况 ===")
    
    for file_name in files_to_check:
        file_path = Path(file_name)
        if not file_path.exists():
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否导入了配置
        if 'from config import' in content or 'import config' in content:
            print(f"✅ {file_name} 已导入配置")
        else:
            print(f"❌ {file_name} 未导入配置")
        
        # 检查是否还有硬编码的365
        if re.search(r'\b365\b', content):
            matches = re.finditer(r'\b365\b', content)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                line_content = content.split('\n')[line_num - 1].strip()
                print(f"⚠️ {file_name} 第{line_num}行仍有365: {line_content}")
        else:
            print(f"✅ {file_name} 已清理所有365硬编码")

def main():
    """主函数"""
    print("=== 完整配置修复脚本 ===\n")
    
    print("修复 lstm_cosmic_ray.py...")
    fix_lstm_cosmic_ray()
    
    print("\n修复 data_processor.py...")
    fix_data_processor()
    
    print("\n验证修复结果...")
    verify_config_usage()
    
    print("\n=== 修复完成 ===")
    print("请运行以下命令验证:")
    print("python check_hardcoded.py")

if __name__ == "__main__":
    main()