import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import logging
import data_processor
# import warnings
# warnings.filterwarnings('ignore')


# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class CosmicRayDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class lstm_model(nn.Module):
    """
    lstm_model 是一个用于时间序列回归预测的神经网络模型，核心结构如下：

    1. LSTM层：{num_layers}层堆叠，每层{hidden_size}个隐藏单元
    2. 全连接层：将LSTM对365天历史数据的最终抽象表示映射到宇宙线强度预测值
    - LSTM的最后一个隐藏状态包含了对整个时间序列的压缩表示
    - 先降维到{hidden_size//2}，经过激活函数和Dropout防止过拟合，最后输出1个预测值

    输入：x，形状为(batch_size, 365, 6) - 365天的6个参数（5个太阳参数+1个宇宙线参数）
    输出：预测值，形状为(batch_size, 1) - 宇宙线强度
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.05):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
        
        # 对LSTM输出的最后一天隐藏状态做归一化，
        # 虽然输入数据已经做了归一化，但LSTM输出的隐藏状态在训练过程中分布可能会发生变化，
        # LayerNorm可以动态调整，进一步提升模型鲁棒性。
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # fc:全连接层(Fully Connected layer), 
        # Sequential是一个容器，
        # 用来把多个神经网络层（比如线性层、激活函数、Dropout等）按顺序组合在一起，
        # 形成一个“流水线”式的网络结构
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), # 第一个线性层：降维
            # nn.BatchNorm1d(hidden_size // 2), # 对全连接层的输出做归一化(按需使用)
            # nn.LayerNorm(hidden_size//2), # 对全连接层的输出做归一化(按需使用)
            nn.ReLU(), # 先用最常用的ReLU，等后续需要调优时再尝试其他激活函数
            # nn.LeakyReLU(0.01),  # 或者 nn.GELU()等别的激活函数
            nn.Dropout(dropout),  # 防止过拟合
            nn.Linear(hidden_size // 2, output_size) # 第二个线性层：输出一个数（预测值）
        )
    
    # 定义数据如何流动
    # 训练和预测时，PyTorch会自动调用 forward，
    # 只需要定义好它，模型就知道怎么处理输入数据了。
    def forward(self, x):
        # x = self.input_norm(x) # x:(batch, seq_len, feat)
        lstm_out, _ = self.lstm(x) # 1. 先经过LSTM层
        h = lstm_out[:, -1, :] # 2. 取最后一天的隐藏状态
        h = self.layer_norm(h) # 3. 做归一化
        output = self.fc(h) # 4. 经过全连接层，输出预测值
        return output


def train_model(model, train_loader, val_loader, num_epochs=1000):
    """
    使用MSE损失函数和Adam优化器
    支持学习率调度（ReduceLROnPlateau）和早停机制（Early Stopping）
    梯度裁剪防止梯度爆炸
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    ################# 可调超参数 ########################################################################################
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0) 
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
    # )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-4,
        steps_per_epoch=len(train_loader), epochs=num_epochs
    )

    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"开始训练模型，使用设备: {device}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_corrected_model.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        if patience_counter >= 30:  # 增加早停patience
            print(f"早停在第 {epoch+1} 轮")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_corrected_model.pth'))
    
    return train_losses, val_losses

