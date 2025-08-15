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

def calculate_metrics_and_stats(cosmic_data, test_actuals, test_predictions, 
                               extended_pred_dates=None, extended_predictions=None):
    """计算详细的评估指标和统计信息"""
    print(f"\n=== 详细评估结果 ===")
    
    # 测试集指标
    mse = mean_squared_error(test_actuals, test_predictions)
    mae = mean_absolute_error(test_actuals, test_predictions)
    r2 = r2_score(test_actuals, test_predictions)
    mape = np.mean(np.abs((test_actuals - test_predictions) / test_actuals)) * 100
    mre = np.mean((test_actuals - test_predictions) / test_actuals) * 100
    
    print(f"测试集性能:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²: {r2:.6f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  MRE: {mre:.2f}%")
    
    # 如果有扩展预测，计算重叠期间的指标
    if extended_pred_dates is not None and extended_predictions is not None:
        obs_dates = cosmic_data['date YYYY-MM-DD']
        obs_flux = cosmic_data['helium_flux m^-2sr^-1s^-1GV^-1']
        
        # 找到重叠的日期
        aligned_obs = []
        aligned_pred = []
        
        for i, pred_date in enumerate(extended_pred_dates):
            matching_obs = cosmic_data[cosmic_data['date YYYY-MM-DD'] == pred_date]
            if not matching_obs.empty:
                aligned_obs.append(matching_obs['helium_flux m^-2sr^-1s^-1GV^-1'].iloc[0])
                aligned_pred.append(extended_predictions[i])
        
        if len(aligned_obs) > 0:
            overlap_mse = mean_squared_error(aligned_obs, aligned_pred)
            overlap_mae = mean_absolute_error(aligned_obs, aligned_pred)
            overlap_r2 = r2_score(aligned_obs, aligned_pred)
            overlap_mape = np.mean(np.abs((np.array(aligned_obs) - np.array(aligned_pred)) / np.array(aligned_obs))) * 100
            overlap_mre = np.mean((np.array(aligned_obs) - np.array(aligned_pred)) / np.array(aligned_obs)) * 100
            
            print(f"\n完整预测重叠期间性能:")
            print(f"  对比数据点数: {len(aligned_obs)}")
            print(f"  MSE: {overlap_mse:.6f}")
            print(f"  MAE: {overlap_mae:.6f}")
            print(f"  R²: {overlap_r2:.6f}")
            print(f"  MAPE: {overlap_mape:.2f}%")
            print(f"  MRE: {overlap_mre:.2f}%")
        
        # 统计信息
        print(f"\n=== 预测统计信息 ===")
        print(f"观测数据统计:")
        print(f"  平均值: {obs_flux.mean():.2f}")
        print(f"  标准差: {obs_flux.std():.2f}")
        print(f"  最小值: {obs_flux.min():.2f}")
        print(f"  最大值: {obs_flux.max():.2f}")
        
        print(f"\n扩展预测统计:")
        print(f"  预测数据点数: {len(extended_predictions)}")
        print(f"  平均值: {np.mean(extended_predictions):.2f}")
        print(f"  标准差: {np.std(extended_predictions):.2f}")
        print(f"  最小值: {np.min(extended_predictions):.2f}")
        print(f"  最大值: {np.max(extended_predictions):.2f}")
        
        # 未来预测统计
        future_start = obs_dates.iloc[-1]
        future_predictions = [p for i, p in enumerate(extended_predictions) 
                            if extended_pred_dates[i] > future_start]
        
        if len(future_predictions) > 0:
            print(f"\n未来预测统计 (2019年后):")
            print(f"  未来预测点数: {len(future_predictions)}")
            print(f"  平均值: {np.mean(future_predictions):.2f}")
            print(f"  标准差: {np.std(future_predictions):.2f}")
            print(f"  最小值: {np.min(future_predictions):.2f}")
            print(f"  最大值: {np.max(future_predictions):.2f}")


def save_complete_results(test_dates, test_actuals, test_predictions, 
                         extended_pred_dates=None, extended_predictions=None):
    """保存完整的预测结果"""
    # 保存测试集结果
    test_results = pd.DataFrame({
        'date': test_dates,
        'actual_flux': test_actuals,
        'predicted_flux': test_predictions,
        'absolute_error': np.abs(test_actuals - test_predictions),
        'relative_error': np.abs(test_actuals - test_predictions) / test_actuals * 100
    })
    test_results.to_csv('测试集预测结果.csv', index=False)
    print(f"\n测试集结果已保存到 '测试集预测结果.csv'")
    
    # 保存扩展预测结果
    if extended_pred_dates is not None and extended_predictions is not None:
        extended_results = pd.DataFrame({
            'date': extended_pred_dates,
            'predicted_flux': extended_predictions
        })
        extended_results.to_csv('宇宙线预测结果_2011_2025.csv', index=False)
        print(f"扩展预测结果已保存到 '宇宙线预测结果_2011_2025.csv'")


def save_metrics_to_txt(train_actuals, train_predictions, test_actuals, test_predictions, filename='model_metrics.txt'):
    """保存训练集和测试集的主要性能指标到txt文件"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np
    
    # 训练集指标
    train_mse = mean_squared_error(train_actuals, train_predictions)
    train_mae = mean_absolute_error(train_actuals, train_predictions)
    train_r2 = r2_score(train_actuals, train_predictions)
    train_mre = np.mean((train_actuals - train_predictions) / train_actuals) * 100
    train_mape = np.mean(np.abs((train_actuals - train_predictions) / train_actuals)) * 100
    
    # 测试集指标
    test_mse = mean_squared_error(test_actuals, test_predictions)
    test_mae = mean_absolute_error(test_actuals, test_predictions)
    test_r2 = r2_score(test_actuals, test_predictions)
    test_mre = np.mean((test_actuals - test_predictions) / test_actuals) * 100
    test_mape = np.mean(np.abs((test_actuals - test_predictions) / test_actuals)) * 100
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('模型性能指标\n')
        f.write('--------------------------\n')
        f.write(f'训练集样本数: {len(train_actuals)}\n')
        f.write(f'训练集 MSE: {train_mse:.6f}\n')
        f.write(f'训练集 MAE: {train_mae:.6f}\n')
        f.write(f'训练集 R²: {train_r2:.6f}\n')
        f.write(f'训练集 MAPE: {train_mape:.2f}%\n')
        f.write(f'训练集 MRE: {train_mre:.2f}%\n')
        f.write('--------------------------\n')
        f.write(f'测试集样本数: {len(test_actuals)}\n')
        f.write(f'测试集 MSE: {test_mse:.6f}\n')
        f.write(f'测试集 MAE: {test_mae:.6f}\n')
        f.write(f'测试集 R²: {test_r2:.6f}\n')
        f.write(f'测试集 MAPE: {test_mape:.2f}%\n')
        f.write(f'测试集 MRE: {test_mre:.2f}%\n')
        f.write('--------------------------\n')
        f.write('说明: MRE为平均相对误差(可为负)，MAPE为平均绝对百分比误差。\n')
        f.write('所有指标均为自动保存，无需手动复制。\n')
