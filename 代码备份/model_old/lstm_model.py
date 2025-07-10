import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import os

class CosmicRayLSTM(nn.Module):
    """
    用于宇宙线通量预测的LSTM神经网络
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        """
        初始化LSTM网络
        
        Args:
            input_size: 输入特征数（太阳参数数量）
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            dropout: Dropout比例
        """
        super(CosmicRayLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        
        # 激活函数和正则化
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, sequence_length, input_size)
            
        Returns:
            输出张量 (batch_size, 1)
        """
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 全连接层
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class ModelTrainer:
    """
    LSTM模型训练器
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module) -> float:
        """
        训练一个epoch
        
        Args:
            dataloader: 数据加载器
            optimizer: 优化器
            criterion: 损失函数
            
        Returns:
            平均损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, dataloader: DataLoader, criterion: nn.Module) -> float:
        """
        验证模型
        
        Args:
            dataloader: 验证数据加载器
            criterion: 损失函数
            
        Returns:
            平均验证损失
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 100, learning_rate: float = 0.001,
              patience: int = 15, save_path: str = None) -> None:
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            learning_rate: 学习率
            patience: 早停耐心值
            save_path: 模型保存路径
        """
        # 优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 早停参数
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("开始训练...")
        print(f"设备: {self.device}")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            
            # 验证
            val_loss = self.validate(val_loader, criterion)
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # 保存最佳模型
                if save_path:
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss
                    }, save_path)
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f'早停于第 {epoch+1} 轮，最佳验证损失: {best_val_loss:.6f}')
                break
        
        print("训练完成!")
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        模型预测
        
        Args:
            X: 输入数据
            
        Returns:
            预测结果
        """
        self.model.eval()
        X = X.to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X)
            
        return predictions.cpu().numpy().squeeze()
    
    def plot_training_history(self, save_path: str = None) -> None:
        """
        绘制训练历史
        
        Args:
            save_path: 图片保存路径
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='训练损失', alpha=0.7)
        plt.plot(self.val_losses, label='验证损失', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('训练历史')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    计算评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        评估指标字典
    """
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # 相关系数
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    
    # R²系数
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Correlation': correlation
    }