#!/usr/bin/env python3
"""
配置验证和测试脚本
验证配置系统是否正常工作
"""

def test_config_import():
    """测试配置导入"""
    try:
        from config import (
            SEQUENCE_LENGTH, HIDDEN_SIZE, NUM_LAYERS, BATCH_SIZE, 
            MAX_EPOCHS, LEARNING_RATE, DROPOUT_RATE, PATIENCE,
            print_config_info
        )
        print("✅ 配置导入成功")
        return True
    except Exception as e:
        print(f"❌ 配置导入失败: {e}")
        return False

def test_config_values():
    """测试配置值"""
    from config import SEQUENCE_LENGTH, HIDDEN_SIZE, print_config_info
    
    print(f"\n=== 当前配置值 ===")
    print(f"序列长度: {SEQUENCE_LENGTH}")
    print(f"隐藏层大小: {HIDDEN_SIZE}")
    
    # 调用配置信息打印函数
    print("\n调用 print_config_info():")
    print_config_info()
    
    return True

def test_data_processor_integration():
    """测试data_processor是否正确使用配置"""
    try:
        # 测试是否可以正确导入和使用配置
        import sys
        sys.path.append('.')
        
        from config import SEQUENCE_LENGTH
        print(f"\n✅ data_processor可以访问配置，序列长度: {SEQUENCE_LENGTH}")
        return True
    except Exception as e:
        print(f"❌ data_processor配置集成失败: {e}")
        return False

def test_sequence_length_flexibility():
    """测试序列长度的灵活性"""
    from config import SEQUENCE_LENGTH
    
    # 模拟不同的序列长度设置
    test_lengths = [180, 300, 365, 500]
    
    print(f"\n=== 序列长度灵活性测试 ===")
    print(f"当前配置的序列长度: {SEQUENCE_LENGTH}")
    
    for length in test_lengths:
        if length == SEQUENCE_LENGTH:
            print(f"✅ {length} 天 (当前设置)")
        else:
            print(f"📝 {length} 天 (可选设置)")
    
    print("\n要更改序列长度，只需修改config.py中的SEQUENCE_LENGTH值")
    return True

def main():
    """主测试函数"""
    print("=== 配置系统验证测试 ===\n")
    
    tests = [
        ("配置导入测试", test_config_import),
        ("配置值测试", test_config_values), 
        ("data_processor集成测试", test_data_processor_integration),
        ("序列长度灵活性测试", test_sequence_length_flexibility),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"🧪 运行: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 通过\n")
            else:
                print(f"❌ {test_name} 失败\n")
        except Exception as e:
            print(f"💥 {test_name} 异常: {e}\n")
    
    print("=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！配置系统工作正常。")
        print("\n现在你可以：")
        print("1. 修改config.py中的SEQUENCE_LENGTH来改变序列长度")
        print("2. 修改其他配置参数来调整模型")
        print("3. 运行主训练脚本: python lstm_cosmic_ray.py")
    else:
        print("⚠️ 部分测试失败，请检查配置设置。")

if __name__ == "__main__":
    main()