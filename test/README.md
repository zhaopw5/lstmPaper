# 测试文件夹

这个文件夹包含用于学习和测试的脚本。

## 文件说明

### resample_demo.py
- **用途**: 演示pandas的resample函数用法
- **内容**: 
  - 时间序列重采样的基本操作
  - 不同聚合函数的使用
  - offset参数的作用（特别是中午12点基准）
  - 缺失数据处理和插值
  - 可视化对比

### 运行方法
```bash
cd test
python resample_demo.py
```

### 依赖库
确保安装了以下Python库：
- pandas
- numpy  
- matplotlib

### 输出
- 控制台输出：各种resample操作的结果
- 图片文件：`resample_demo.png`（保存在test文件夹中）
