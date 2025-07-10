#!/usr/bin/env python3
"""
宇宙线LSTM预测项目 - 简单演示脚本
这个脚本演示了如何使用训练好的模型进行预测
"""

import sys
import os
sys.path.append('/home/phil/Files/lstmPaper_v250618/model')

def check_data_files():
    """检查数据文件是否存在"""
    solar_path = '/home/phil/Files/lstmPaper_v250618/data/solar_physics_data_1980_extend_smoothed.csv'
    cosmic_path = '/home/phil/Files/lstmPaper_v250618/data/raw_data/ams/helium.csv'
    
    print("=== 数据文件检查 ===")
    print(f"太阳参数数据: {os.path.exists(solar_path)} - {solar_path}")
    print(f"宇宙线数据: {os.path.exists(cosmic_path)} - {cosmic_path}")
    
    if os.path.exists(solar_path) and os.path.exists(cosmic_path):
        print("✓ 数据文件完整")
        return True
    else:
        print("✗ 数据文件缺失")
        return False

def check_dependencies():
    """检查依赖包是否安装"""
    print("\n=== 依赖包检查 ===")
    packages = ['torch', 'pandas', 'numpy', 'sklearn', 'matplotlib']
    
    missing_packages = []
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - 缺失")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n请安装缺失的包:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("✓ 所有依赖包已安装")
        return True

def show_project_structure():
    """显示项目结构"""
    print("\n=== 项目结构 ===")
    base_path = '/home/phil/Files/lstmPaper_v250618/model'
    
    files = [
        'data_processor.py',
        'lstm_model.py', 
        'train_cosmic_ray_lstm.py',
        'requirements.txt',
        'README.md'
    ]
    
    for file in files:
        file_path = os.path.join(base_path, file)
        exists = os.path.exists(file_path)
        status = "✓" if exists else "✗"
        print(f"{status} {file}")

def run_quick_demo():
    """运行快速演示"""
    print("\n=== 快速演示 ===")
    
    try:
        # 检查是否能导入模块
        from data_processor import CosmicRayDataProcessor
        print("✓ 数据处理模块导入成功")
        
        from lstm_model import CosmicRayLSTM
        print("✓ LSTM模型模块导入成功")
        
        # 简单演示数据加载
        solar_path = '/home/phil/Files/lstmPaper_v250618/data/solar_physics_data_1980_extend_smoothed.csv'
        cosmic_path = '/home/phil/Files/lstmPaper_v250618/data/raw_data/ams/helium.csv'
        
        if os.path.exists(solar_path) and os.path.exists(cosmic_path):
            print("✓ 正在测试数据加载...")
            processor = CosmicRayDataProcessor(solar_path, cosmic_path)
            processor.load_data()
            print(f"✓ 数据加载成功 - 太阳数据: {processor.solar_data.shape}, 宇宙线数据: {processor.cosmic_data.shape}")
        else:
            print("✗ 数据文件不存在，跳过数据加载测试")
        
        print("\n✓ 快速演示完成！")
        
    except Exception as e:
        print(f"✗ 演示失败: {e}")

def main():
    """主函数"""
    print("宇宙线LSTM预测项目 - 初学者检查工具")
    print("="*50)
    
    # 检查各项配置
    data_ok = check_data_files()
    deps_ok = check_dependencies()
    show_project_structure()
    
    if data_ok and deps_ok:
        run_quick_demo()
        print("\n" + "="*50)
        print("🎉 恭喜！你的环境配置完整")
        print("现在可以运行完整的训练脚本:")
        print("python train_cosmic_ray_lstm.py")
    else:
        print("\n" + "="*50)
        print("❌ 环境配置不完整，请先解决上述问题")
        print("参考 README.md 获取详细说明")

if __name__ == "__main__":
    main()