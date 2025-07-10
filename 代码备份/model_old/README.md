# 宇宙线LSTM预测项目
这个项目使用LSTM神经网络预测宇宙线通量，基于太阳参数的历史数据。

## 项目结构
```
/home/phil/Files/lstmPaper_v250618/model/
├── data_processor.py          # 数据处理模块
├── lstm_model.py             # LSTM模型定义
├── train_cosmic_ray_lstm.py  # 主训练脚本
├── requirements.txt          # 依赖包列表
└── results/                  # 训练结果保存目录
```

## 安装依赖
```bash
pip install torch pandas numpy scikit-learn matplotlib
```

## 快速开始

### 1. 检查数据
确保以下数据文件存在：
- 太阳参数数据: `/home/phil/Files/lstmPaper_v250618/data/solar_physics_data_1980_extend_smoothed.csv`
- 宇宙线数据: `/home/phil/Files/lstmPaper_v250618/data/raw_data/ams/helium.csv`

### 2. 运行训练
```bash
cd /home/phil/Files/lstmPaper_v250618/model
python train_cosmic_ray_lstm.py
```

## 项目详解

### 数据说明
- **输入**: 365天 × 5个太阳参数 (HMF, wind_speed, HCS_tilt, polarity, SSN)
- **输出**: 1个宇宙线通量值 (rigidity_min GV = 2.97)
- **样本数**: 约3085个 (2011年5月20日到2019年10月29日)

### 模型架构
- **LSTM层**: 2层，隐藏单元128个
- **全连接层**: 降维 + 输出层
- **正则化**: Dropout (0.2)
- **优化器**: Adam
- **损失函数**: MSE

### 训练策略
- **数据分割**: 70%训练，15%验证，15%测试
- **批大小**: 32
- **学习率**: 0.001
- **早停**: 验证损失15轮不改善时停止

## 结果解读

### 评估指标
- **MSE**: 均方误差
- **RMSE**: 均方根误差  
- **MAE**: 平均绝对误差
- **R²**: 决定系数
- **Correlation**: 相关系数

### 输出文件
- `best_model.pth`: 最佳模型权重
- `training_history.png`: 训练损失曲线
- `train_predictions.png`: 训练集预测结果
- `val_predictions.png`: 验证集预测结果
- `test_predictions.png`: 测试集预测结果

## 初学者指南

### 1. 理解时间序列预测
这是一个时间序列回归问题：
- 用过去365天的太阳活动预测今天的宇宙线通量
- 时间顺序很重要，不能随机打乱数据

### 2. LSTM的优势
- 能记住长期依赖关系
- 适合处理序列数据
- 能捕捉太阳活动对宇宙线的延迟影响

### 3. 数据预处理的重要性
- 标准化：让不同量级的特征在同一尺度
- 时间对齐：确保输入和输出时间匹配正确
- 缺失值处理：保证数据完整性

### 4. 模型调优建议
- 增加隐藏层大小可能提高性能，但要防止过拟合
- 调整序列长度（目前365天）可能影响预测效果
- 添加更多太阳参数可能改善预测精度

### 5. 结果分析
- R² > 0.8 表示模型效果较好
- 训练损失和验证损失应该一起下降
- 预测值与真实值的散点图应该接近对角线

## 常见问题

### Q: 训练很慢怎么办？
A: 
- 减小batch_size
- 减少LSTM层数或隐藏单元数
- 使用GPU加速（如果有的话）

### Q: 模型过拟合怎么办？
A:
- 增加dropout比例
- 减少模型复杂度
- 增加训练数据
- 使用正则化

### Q: 预测效果不好怎么办？
A:
- 检查数据质量和预处理
- 尝试不同的网络架构
- 调整超参数
- 增加更多特征

## 进一步学习
1. 学习PyTorch基础
2. 了解时间序列分析
3. 研究LSTM原理
4. 学习深度学习优化技巧
5. 探索其他序列模型（GRU, Transformer等）