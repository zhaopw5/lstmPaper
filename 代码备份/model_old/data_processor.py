import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import torch
from typing import Tuple, List

class CosmicRayDataProcessor:
    """
    宇宙线预测数据处理器
    用于处理太阳参数和宇宙线通量数据，构建LSTM训练数据集
    """
    
    def __init__(self, solar_data_path: str, cosmic_ray_path: str, 
                 sequence_length: int = 365, target_rigidity: float = 2.97):
        """
        初始化数据处理器
        
        Args:
            solar_data_path: 太阳参数数据文件路径
            cosmic_ray_path: 宇宙线数据文件路径
            sequence_length: 输入序列长度（天数）
            target_rigidity: 目标刚度值
        """
        self.solar_data_path = solar_data_path
        self.cosmic_ray_path = cosmic_ray_path
        self.sequence_length = sequence_length
        self.target_rigidity = target_rigidity
        
        # 标准化器
        self.solar_scaler = StandardScaler()
        self.cosmic_scaler = StandardScaler()
        
        # 数据存储
        self.solar_data = None
        self.cosmic_data = None
        self.processed_data = None
        
    def load_data(self) -> None:
        """加载原始数据"""
        print("正在加载数据...")
        
        # 加载太阳参数数据
        self.solar_data = pd.read_csv(self.solar_data_path)
        self.solar_data['date'] = pd.to_datetime(self.solar_data['date'])
        self.solar_data = self.solar_data.sort_values('date').reset_index(drop=True)
        
        # 加载宇宙线数据
        self.cosmic_data = pd.read_csv(self.cosmic_ray_path)
        self.cosmic_data['date YYYY-MM-DD'] = pd.to_datetime(self.cosmic_data['date YYYY-MM-DD'])
        
        # 筛选特定刚度的数据
        self.cosmic_data = self.cosmic_data[
            self.cosmic_data['rigidity_min GV'] == self.target_rigidity
        ].copy()
        self.cosmic_data = self.cosmic_data.sort_values('date YYYY-MM-DD').reset_index(drop=True)
        
        print(f"太阳数据形状: {self.solar_data.shape}")
        print(f"宇宙线数据形状: {self.cosmic_data.shape}")
        print(f"太阳数据时间范围: {self.solar_data['date'].min()} 到 {self.solar_data['date'].max()}")
        print(f"宇宙线数据时间范围: {self.cosmic_data['date YYYY-MM-DD'].min()} 到 {self.cosmic_data['date YYYY-MM-DD'].max()}")
        
    def create_sequences(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        创建LSTM训练序列
        
        Returns:
            X: 输入序列 (n_samples, sequence_length, n_features)
            y: 输出标签 (n_samples,)
            dates: 对应的日期列表
        """
        print("正在创建训练序列...")
        
        # 太阳参数特征列
        solar_features = ['HMF', 'wind_speed', 'HCS_tilt', 'polarity', 'SSN']
        
        X_list = []
        y_list = []
        dates_list = []
        
        # 为每个宇宙线观测日期创建对应的输入序列
        for idx, row in self.cosmic_data.iterrows():
            cosmic_date = row['date YYYY-MM-DD']
            cosmic_flux = row['helium_flux m^-2sr^-1s^-1GV^-1']
            
            # 计算需要的太阳数据日期范围：cosmic_date前365天
            start_date = cosmic_date - timedelta(days=self.sequence_length)
            end_date = cosmic_date - timedelta(days=1)
            
            # 提取对应日期范围的太阳数据
            solar_subset = self.solar_data[
                (self.solar_data['date'] >= start_date) & 
                (self.solar_data['date'] <= end_date)
            ].copy()
            
            # 检查是否有足够的数据
            if len(solar_subset) == self.sequence_length:
                # 提取特征数据
                solar_features_data = solar_subset[solar_features].values
                
                X_list.append(solar_features_data)
                y_list.append(cosmic_flux)
                dates_list.append(cosmic_date.strftime('%Y-%m-%d'))
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"成功创建 {len(X)} 个训练样本")
        print(f"输入维度: {X.shape}")
        print(f"输出维度: {y.shape}")
        
        return X, y, dates_list
    
    def normalize_data(self, X: np.ndarray, y: np.ndarray, 
                      fit_scalers: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据标准化
        
        Args:
            X: 输入数据
            y: 输出数据
            fit_scalers: 是否拟合标准化器
            
        Returns:
            标准化后的X, y
        """
        print("正在进行数据标准化...")
        
        # 重塑X以便标准化
        n_samples, seq_len, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        
        if fit_scalers:
            # 拟合并转换
            X_normalized = self.solar_scaler.fit_transform(X_reshaped)
            y_normalized = self.cosmic_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        else:
            # 仅转换
            X_normalized = self.solar_scaler.transform(X_reshaped)
            y_normalized = self.cosmic_scaler.transform(y.reshape(-1, 1)).flatten()
        
        # 重塑回原来的形状
        X_normalized = X_normalized.reshape(n_samples, seq_len, n_features)
        
        return X_normalized, y_normalized
    
    def inverse_transform_y(self, y_normalized: np.ndarray) -> np.ndarray:
        """反标准化输出数据"""
        return self.cosmic_scaler.inverse_transform(y_normalized.reshape(-1, 1)).flatten()
    
    def prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        完整的数据准备流程
        
        Returns:
            X_tensor: PyTorch张量格式的输入数据
            y_tensor: PyTorch张量格式的输出数据
            dates: 日期列表
        """
        # 加载数据
        self.load_data()
        
        # 创建序列
        X, y, dates = self.create_sequences()
        
        # 标准化
        X_normalized, y_normalized = self.normalize_data(X, y, fit_scalers=True)
        
        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X_normalized)
        y_tensor = torch.FloatTensor(y_normalized)
        
        print("数据准备完成!")
        print(f"最终数据形状 - X: {X_tensor.shape}, y: {y_tensor.shape}")
        
        return X_tensor, y_tensor, dates

def split_time_series(X: torch.Tensor, y: torch.Tensor, dates: List[str], 
                     train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple:
    """
    时间序列数据分割
    
    Args:
        X: 输入数据
        y: 输出数据
        dates: 日期列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        
    Returns:
        训练集、验证集、测试集的数据和日期
    """
    n_samples = len(X)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    # 时间序列分割（不打乱顺序）
    X_train = X[:train_size]
    y_train = y[:train_size]
    dates_train = dates[:train_size]
    
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    dates_val = dates[train_size:train_size + val_size]
    
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    dates_test = dates[train_size + val_size:]
    
    print(f"数据分割完成:")
    print(f"训练集: {len(X_train)} 样本")
    print(f"验证集: {len(X_val)} 样本") 
    print(f"测试集: {len(X_test)} 样本")
    
    return (X_train, y_train, dates_train), (X_val, y_val, dates_val), (X_test, y_test, dates_test)