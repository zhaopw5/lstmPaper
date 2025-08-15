# LSTM宇宙线预测模型项目

## 项目概述
输入是solar + helium数据，采用递归预测方法，预测值写入字典供后续递归使用。

## 脚本功能说明

### 主程序
- **lstm_cosmic_ray.py** - 主程序入口，协调各模块完成完整的训练和预测流程

### 数据处理模块
- **data_processor.py** - 数据加载、预处理、序列创建和归一化功能

### 模型相关
- **models.py** - LSTM模型定义、数据集类和训练函数

### 预测模块
- **predictions.py** - 扩展预测功能，包括日期生成和递归预测

### 可视化模块
- **visualization.py** - 绘制训练损失、预测结果等综合可视化图表

### 评估模块
- **evaluation.py** - 模型性能评估、指标计算和结果保存功能

## 工作流程
1. 数据加载与预处理 (data_processor)
2. 模型训练 (models)
3. 模型评估 (evaluation)
4. 扩展预测 (predictions)
5. 结果可视化 (visualization)
6. 结果保存 (evaluation)