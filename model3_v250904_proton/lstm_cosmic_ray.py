import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
from pathlib import Path

# 导入配置和数据处理模块
from config import (
    SEQUENCE_LENGTH, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_RATE,
    BATCH_SIZE, MAX_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    PATIENCE, GRAD_CLIP_NORM, RESULTS_DIR,
    BEST_MODEL_PATH, MODEL_SUMMARY_PATH, TRAINING_HISTORY_PATH,
    MODEL_METRICS_PATH, TEST_PREDICTIONS_PATH, EXTENDED_PREDICTIONS_PATH,
    MODEL_SAVE_PATH, print_config_info
)
from data_processor import (
    load_and_check_data, create_sequences, split_data_three_way, 
    normalize_data_three_way, SOLAR_PARAMETERS, RIGIDITY_VALUES
)

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
    lstm_model 是一个用于多刚度宇宙线时间序列回归预测的神经网络模型，核心结构如下：

    1. LSTM层：多层堆叠，每层具有可配置的隐藏单元数
    2. 全连接层：将LSTM对序列历史数据的最终抽象表示映射到多个刚度的宇宙线强度预测值
    - LSTM的最后一个隐藏状态包含了对整个时间序列的压缩表示
    - 先降维，经过激活函数和Dropout防止过拟合，最后输出多个刚度的预测值

    输入：x，形状为(batch_size, sequence_length, input_size) - 配置长度的多个参数
    输出：预测值，形状为(batch_size, output_size) - 多个刚度的宇宙线强度
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


def train_model(model, train_loader, val_loader, num_epochs=MAX_EPOCHS):
    """
    使用MSE损失函数和Adam优化器
    支持学习率调度（OneCycleLR）和早停机制（Early Stopping）
    梯度裁剪防止梯度爆炸
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) 
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader), epochs=num_epochs
    )

    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_path = Path(BEST_MODEL_PATH).resolve()
    
    print(f"开始训练模型，使用设备: {device}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)  # 多维输出，不需要squeeze
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            
            optimizer.step()
            # OneCycleLR 需要按batch更新
            scheduler.step()
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), str(best_path))
            print(f"保存当前最佳模型到: {best_path}")
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        if patience_counter >= PATIENCE:
            print(f"早停在第 {epoch+1} 轮")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(str(best_path)))
    
    return train_losses, val_losses


def evaluate_on_test_set(model, test_loader, scaler_y):
    """在独立测试集上评估模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    test_predictions = []
    test_actuals = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            test_predictions.extend(outputs.cpu().numpy())
            test_actuals.extend(y_batch.numpy())

    # 反归一化
    test_predictions = scaler_y.inverse_transform(np.array(test_predictions).reshape(-1, len(RIGIDITY_VALUES)))
    test_actuals = scaler_y.inverse_transform(np.array(test_actuals).reshape(-1, len(RIGIDITY_VALUES)))

    return test_predictions, test_actuals


def create_prediction_dates(start_date, end_date):
    """创建预测日期序列（返回pandas Timestamp列表，确保与数据键类型一致）"""
    return list(pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq='D'))


#############################################################
######--------------->>> 递归预测 <<<------------------######
#############################################################

def predict_cosmic_ray_extended(model, solar_data, cosmic_data, prediction_dates, scaler_X, scaler_y, sequence_length=SEQUENCE_LENGTH):
    """
    递归式长期预测：
    - 观测期内，输入序列氦通量用真实数据
    - 观测期外，输入序列氦通量用已预测值递归推进
    """
    print(f"\n=== 扩展预测（递归式，序列长度={sequence_length}天）：{len(prediction_dates)} 个日期 ===")

    proton_flux_cols = [
        f'proton_{rigidity}GV' for rigidity in RIGIDITY_VALUES
        if f'proton_{rigidity}GV' in cosmic_data.columns
    ]
    features = SOLAR_PARAMETERS + proton_flux_cols

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_predictions = []
    valid_dates = []

    # cosmic_data按日期排序
    cosmic_data = cosmic_data.sort_values('date YYYY-MM-DD').reset_index(drop=True)

    # 构造多刚度氦通量时间序列（初始为观测数据）
    proton_flux_dict = {}
    for col in proton_flux_cols:
        proton_flux_series = cosmic_data[['date YYYY-MM-DD', col]].copy()
        proton_flux_dict[col] = dict(zip(proton_flux_series['date YYYY-MM-DD'], proton_flux_series[col]))

    for i, target_date in enumerate(prediction_dates):
        if (i + 1) % 1000 == 0:
            print(f"已处理 {i + 1}/{len(prediction_dates)} 个日期")

        # 计算需要的日期范围
        start_date = target_date - timedelta(days=sequence_length)
        end_date = target_date - timedelta(days=1)
        expected_dates = pd.date_range(start=start_date, end=end_date, freq='D')

        input_rows = []
        for date_i in expected_dates:
            solar_mask = solar_data['date'] == date_i

            # 获取所有刚度的氦通量
            proton_flux_values = []
            missing_flux = False

            for col in proton_flux_cols:
                if date_i in proton_flux_dict[col]:
                    proton_flux_values.append(proton_flux_dict[col][date_i])
                else:
                    missing_flux = True
                    break

            if solar_mask.sum() == 1 and not missing_flux:
                solar_row = solar_data[solar_mask][SOLAR_PARAMETERS].iloc[0].values
                input_rows.append(np.concatenate([solar_row, proton_flux_values]))
            else:
                input_rows = []
                break

        if len(input_rows) == sequence_length:
            solar_sequence_flat = np.array(input_rows).reshape(-1, len(features))
            solar_sequence_scaled_flat = scaler_X.transform(solar_sequence_flat)
            solar_sequence_scaled = solar_sequence_scaled_flat.reshape(1, sequence_length, len(features))

            X_tensor = torch.FloatTensor(solar_sequence_scaled).to(device)
            with torch.no_grad():
                pred_scaled = model(X_tensor).cpu().numpy()
                pred = scaler_y.inverse_transform(pred_scaled.reshape(1, -1)).flatten()

            all_predictions.append(pred)
            valid_dates.append(target_date)

            # 如果该日期没有观测值，则将预测值加入序列，供后续递归
            for j, col in enumerate(proton_flux_cols):
                if target_date not in proton_flux_dict[col]:
                    proton_flux_dict[col][target_date] = pred[j]

    print(f"成功预测了 {len(all_predictions)} 个数据点（递归式扩展）")
    return valid_dates, np.array(all_predictions)


# 统一在反归一化空间计算评估指标，并带R²一致性自检

def compute_metrics_on_original(y_true_orig, y_pred_orig, rigidity_names, y_true_scaled=None, y_pred_scaled=None):
    # 反归一化选择
    if y_true_scaled is not None and y_pred_scaled is not None:
        y_true = y_true_scaled
        y_pred = y_pred_scaled
    else:
        y_true = y_true_orig
        y_pred = y_pred_orig

    results = []
    for i, name in enumerate(rigidity_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        mse = mean_squared_error(yt, yp)
        mae = mean_absolute_error(yt, yp)
        mape = float(np.mean(np.abs((yt - yp) / np.clip(yt, 1e-12, None))) * 100.0)
        r2 = r2_score(yt, yp)

        # 一致性自检：在标准化空间再算一次R²
        if y_true_scaled is not None and y_pred_scaled is not None:
            yt_s = y_true_scaled[:, i]
            yp_s = y_pred_scaled[:, i]
            try:
                r2_scaled = r2_score(yt_s, yp_s)
            except Exception:
                r2_scaled = np.nan
            if np.isfinite(r2_scaled):
                if (r2 < 0 and r2_scaled > 0.2) or (r2 > 0.2 and r2_scaled < 0):
                    print(f"[警告] 指标量纲不一致风险: {name} R2(original)={r2:.3f}, R2(scaled)={r2_scaled:.3f}")

        results.append({
            'name': name,
            'MSE': mse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
        })
    return results


# 新增：R² 诊断与基线对比函数

def _diagnose_r2_and_baselines(y_true, y_pred, rigidity_names):
    print("\n=== R² 诊断（测试期） ===")
    for i, name in enumerate(rigidity_names):
        yt = y_true[:, i].astype(float)
        yp = y_pred[:, i].astype(float)

        var_y = float(np.var(yt, ddof=0))
        mse_model = float(np.mean((yt - yp) ** 2))
        r2 = 1.0 - (mse_model / (var_y + 1e-12))
        mse_const = var_y

        if len(yt) > 1:
            mse_persist = float(np.mean((yt[1:] - yt[:-1]) ** 2))
            skill_persist = 1.0 - (mse_model / (mse_persist + 1e-12))
        else:
            mse_persist, skill_persist = float('nan'), float('nan')

        rmse = float(np.sqrt(mse_model))
        std_y = float(np.sqrt(var_y))
        nrmse_std = rmse / (std_y + 1e-12)
        rng = float(np.max(yt) - np.min(yt))
        nrmse_rng = rmse / (rng + 1e-12)

        std_pred = float(np.std(yp))
        if std_y < 1e-12 or std_pred < 1e-12:
            r = float('nan')
        else:
            r = float(np.corrcoef(yt, yp)[0, 1])

        print(f"  {name}: Var(y)={var_y:.6f}, MSE(model)={mse_model:.6f}, R²={r2:.3f}")
        print(f"    MSE(const-mean)={mse_const:.6f}, MSE(persist)={mse_persist:.6f}, Skill_vs_persist={skill_persist:.3f}")
        print(f"    NRMSE/std={nrmse_std:.2f}, NRMSE/range={nrmse_rng:.3f}, Pearson r={r:.3f}")


def calculate_metrics_and_stats(cosmic_data, test_actuals, test_predictions,
                               extended_pred_dates=None, extended_predictions=None, scaler_y=None):
    """计算详细的评估指标和统计信息（统一原始物理量空间，带一致性自检）"""
    print(f"\n=== 详细评估结果 ===")

    proton_flux_cols = [
        f'proton_{rigidity}GV' for rigidity in RIGIDITY_VALUES
        if f'proton_{rigidity}GV' in cosmic_data.columns
    ]
    rigidity_names = [col.split('_')[-1] for col in proton_flux_cols]

    print(f"测试集性能（按刚度）：")
    print(f"注意：测试期为低波动窗口，R²可能失效，请重点关注 MAPE、Pearson r 和 NRMSE")
    y_true_scaled = scaler_y.transform(test_actuals) if scaler_y is not None else None
    y_pred_scaled = scaler_y.transform(test_predictions) if scaler_y is not None else None

    per_metrics = compute_metrics_on_original(
        y_true_orig=test_actuals,
        y_pred_orig=test_predictions,
        rigidity_names=rigidity_names,
        y_true_scaled=y_true_scaled,
        y_pred_scaled=y_pred_scaled
    )

    # 汇总关键指标
    print(f"\n=== 测试集性能摘要（更适用于低方差期间）===")
    print(f"{'刚度':<8} {'MAPE':<8} {'Pearson_r':<10} {'NRMSE/rng':<10} {'R²':<8} {'备注'}")
    print("-" * 60)

    for i, m in enumerate(per_metrics):
        yt = test_actuals[:, i]
        yp = test_predictions[:, i]
        rng = float(np.max(yt) - np.min(yt))
        rmse = float(np.sqrt(m['MSE']))
        nrmse_rng = rmse / (rng + 1e-12)

        std_pred = float(np.std(yp))
        std_y = float(np.sqrt(np.var(yt, ddof=0)))
        if std_y < 1e-12 or std_pred < 1e-12:
            r = float('nan')
        else:
            r = float(np.corrcoef(yt, yp)[0, 1])

        status = "良好" if (m['MAPE'] < 10 and abs(r) > 0.5) else "一般" if (m['MAPE'] < 15) else "偏差"
        print(f"{m['name']:<8} {m['MAPE']:<8.1f} {r:<10.3f} {nrmse_rng:<10.3f} {m['R2']:<8.3f} {status}")

    for i, m in enumerate(per_metrics):
        yt = test_actuals[:, i]
        yp = test_predictions[:, i]
        mre = float(np.mean((yt - yp) / np.clip(yt, 1e-12, None)) * 100.0)
        print(f"  {m['name']}:")
        print(f"    MSE: {m['MSE']:.6f}, MAE: {m['MAE']:.6f}, R²: {m['R2']:.6f}")
        print(f"    MAPE: {m['MAPE']:.2f}%, MRE: {mre:.2f}%")

    # 测试期R²诊断
    _diagnose_r2_and_baselines(test_actuals, test_predictions, rigidity_names)

    # 如果有扩展预测，计算重叠期间的指标
    if extended_pred_dates is not None and extended_predictions is not None:
        obs_dates = cosmic_data['date YYYY-MM-DD']

        print(f"\n完整预测重叠期间性能（按刚度）：")
        for i, rigidity_col in enumerate(proton_flux_cols):
            rigidity_name = rigidity_col.split('_')[-1]

            aligned_obs = []
            aligned_pred = []
            for j, pred_date in enumerate(extended_pred_dates):
                matching_obs = cosmic_data[cosmic_data['date YYYY-MM-DD'] == pred_date]
                if not matching_obs.empty:
                    aligned_obs.append(matching_obs[rigidity_col].iloc[0])
                    aligned_pred.append(extended_predictions[j, i])

            if len(aligned_obs) > 0:
                aligned_obs = np.array(aligned_obs)
                aligned_pred = np.array(aligned_pred)
                overlap_mse = mean_squared_error(aligned_obs, aligned_pred)
                overlap_mae = mean_absolute_error(aligned_obs, aligned_pred)
                overlap_r2 = r2_score(aligned_obs, aligned_pred)
                overlap_mape = np.mean(np.abs((aligned_obs - aligned_pred) / np.clip(aligned_obs, 1e-12, None))) * 100
                overlap_mre = np.mean((aligned_obs - aligned_pred) / np.clip(aligned_obs, 1e-12, None)) * 100

                print(f"  {rigidity_name} (对比数据点数: {len(aligned_obs)}):")
                print(f"    MSE: {overlap_mse:.6f}, MAE: {overlap_mae:.6f}, R²: {overlap_r2:.6f}")
                print(f"    MAPE: {overlap_mape:.2f}%, MRE: {overlap_mre:.2f}%")

        # 统计信息
        print(f"\n=== 预测统计信息 ===")
        for i, rigidity_col in enumerate(proton_flux_cols):
            rigidity_name = rigidity_col.split('_')[-1]
            obs_flux = cosmic_data[rigidity_col]

            print(f"{rigidity_name} 观测数据统计:")
            print(f"  平均值: {obs_flux.mean():.2f}, 标准差: {obs_flux.std():.2f}")
            print(f"  最小值: {obs_flux.min():.2f}, 最大值: {obs_flux.max():.2f}")

            if extended_predictions is not None:
                pred_flux = extended_predictions[:, i]
                print(f"{rigidity_name} 扩展预测统计:")
                print(f"  预测数据点数: {len(pred_flux)}")
                print(f"  平均值: {np.mean(pred_flux):.2f}, 标准差: {np.std(pred_flux):.2f}")
                print(f"  最小值: {np.min(pred_flux):.2f}, 最大值: {np.max(pred_flux):.2f}")

                # 未来预测统计
                future_start = obs_dates.iloc[-1]
                future_predictions = [p for j, p in enumerate(pred_flux)
                                      if extended_pred_dates[j] > future_start]

                if len(future_predictions) > 0:
                    print(f"{rigidity_name} 未来预测统计:")
                    print(f"  未来预测点数: {len(future_predictions)}")
                    print(f"  平均值: {np.mean(future_predictions):.2f}, 标准差: {np.std(future_predictions):.2f}")
                    print(f"  最小值: {np.min(future_predictions):.2f}, 最大值: {np.max(future_predictions):.2f}")


def save_complete_results(test_dates, test_actuals, test_predictions, 
                         extended_pred_dates=None, extended_predictions=None):
    """保存完整的预测结果到CSV文件。"""
    # 获取所有刚度列名
    proton_flux_cols = [f'proton_{rigidity}GV' for rigidity in RIGIDITY_VALUES]

    # 保存测试集结果 - 每个刚度单独一行
    test_results_list = []
    for i, rigidity_col in enumerate(proton_flux_cols):
        rigidity_name = rigidity_col.split('_')[-1]
        test_actual_i = test_actuals[:, i] if test_actuals.ndim > 1 else test_actuals
        test_pred_i = test_predictions[:, i] if test_predictions.ndim > 1 else test_predictions

        for j, date in enumerate(test_dates):
            actual = test_actual_i[j]
            pred = test_pred_i[j]
            rel_err = (np.abs(actual - pred) / np.clip(actual, 1e-12, None)) * 100
            test_results_list.append({
                'date': date,
                'rigidity': rigidity_name,
                'actual_flux': actual,
                'predicted_flux': pred,
                'absolute_error': np.abs(actual - pred),
                'relative_error': rel_err
            })

    test_results = pd.DataFrame(test_results_list)
    test_out = Path(TEST_PREDICTIONS_PATH).resolve()
    test_results.to_csv(test_out, index=False)
    print(f"\n测试集结果已保存到: {test_out}")

    # 保存扩展预测结果（如果有）
    if extended_pred_dates is not None and extended_predictions is not None:
        extended_results_list = []
        for i, rigidity_col in enumerate(proton_flux_cols):
            rigidity_name = rigidity_col.split('_')[-1]
            for j, date in enumerate(extended_pred_dates):
                extended_results_list.append({
                    'date': date,
                    'rigidity': rigidity_name,
                    'predicted_flux': float(extended_predictions[j, i])
                })
        extended_results = pd.DataFrame(extended_results_list)
        ext_out = Path(EXTENDED_PREDICTIONS_PATH).resolve()
        extended_results.to_csv(ext_out, index=False)
        print(f"扩展预测结果已保存到: {ext_out}")


def save_metrics_to_txt(train_actuals, train_predictions, test_actuals, test_predictions, filename=MODEL_METRICS_PATH):
    """保存训练集和测试集的主要性能指标到txt文件"""
    # 使用在文件顶部导入的 sklearn 指标
    # 计算训练集指标
    try:
        train_mse = mean_squared_error(train_actuals, train_predictions)
        train_mae = mean_absolute_error(train_actuals, train_predictions)
        train_r2 = r2_score(train_actuals, train_predictions)
        train_mre = np.mean((train_actuals - train_predictions) / np.clip(train_actuals, 1e-12, None)) * 100
        train_mape = np.mean(np.abs((train_actuals - train_predictions) / np.clip(train_actuals, 1e-12, None))) * 100
    except Exception:
        train_mse = train_mae = train_r2 = train_mre = train_mape = float('nan')

    # 计算测试集指标
    try:
        test_mse = mean_squared_error(test_actuals, test_predictions)
        test_mae = mean_absolute_error(test_actuals, test_predictions)
        test_r2 = r2_score(test_actuals, test_predictions)
        test_mre = np.mean((test_actuals - test_predictions) / np.clip(test_actuals, 1e-12, None)) * 100
        test_mape = np.mean(np.abs((test_actuals - test_predictions) / np.clip(test_actuals, 1e-12, None))) * 100
    except Exception:
        test_mse = test_mae = test_r2 = test_mre = test_mape = float('nan')

    out_path = Path(filename).resolve()
    with open(out_path, 'w', encoding='utf-8') as f:
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
    print(f"模型性能指标已保存到: {out_path}")


def main():
    """主函数：训练、评估、扩展预测并保存结果"""
    print("=== 完整LSTM宇宙线预测模型（包含训练和扩展预测）===\n")

    # 创建结果目录
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(exist_ok=True)
    print(f"结果将保存到目录: {results_dir.resolve()}")

    # 显示当前配置
    try:
        print_config_info()
    except Exception:
        pass

    # 1. 加载数据
    solar_data, cosmic_data = load_and_check_data()

    # 2. 创建序列（使用配置中的序列长度）
    X, y, dates = create_sequences(solar_data, cosmic_data, sequence_length=SEQUENCE_LENGTH)

    if len(X) == 0:
        print("错误: 没有成功创建任何训练样例！")
        return

    # 3. 划分数据集
    (X_train, X_val, X_test, y_train, y_val, y_test,
     dates_train, dates_val, dates_test) = split_data_three_way(X, y, dates)

    # 4. 归一化（只用训练集拟合归一化器）
    (X_train_scaled, X_val_scaled, X_test_scaled,
     y_train_scaled, y_val_scaled, y_test_scaled,
     scaler_X, scaler_y) = normalize_data_three_way(
        X_train, X_val, X_test, y_train, y_val, y_test)

    # 5. 创建数据加载器
    train_dataset = CosmicRayDataset(X_train_scaled, y_train_scaled)
    val_dataset = CosmicRayDataset(X_val_scaled, y_val_scaled)
    test_dataset = CosmicRayDataset(X_test_scaled, y_test_scaled)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 6. 创建并初始化模型
    input_size = len(SOLAR_PARAMETERS) + len(RIGIDITY_VALUES)
    output_size = len(RIGIDITY_VALUES)
    model = lstm_model(input_size, HIDDEN_SIZE, NUM_LAYERS, output_size, dropout=DROPOUT_RATE)

    print(f"\n模型配置:")
    print(f"  输入特征数: {input_size}")
    print(f"  隐藏层大小: {HIDDEN_SIZE}")
    print(f"  LSTM层数: {NUM_LAYERS}")
    print(f"  序列长度: {SEQUENCE_LENGTH}")
    print(f"  批次大小: {BATCH_SIZE}")

    # 保存模型结构与配置信息
    try:
        with open(MODEL_SUMMARY_PATH, 'w') as f:
            f.write(str(model))
            f.write(f"\n配置参数：\n")
            f.write(f"  序列长度: {SEQUENCE_LENGTH}\n")
            f.write(f"  输入特征数: {input_size}\n")
            f.write(f"  隐藏层大小: {HIDDEN_SIZE}\n")
            f.write(f"  LSTM层数: {NUM_LAYERS}\n")
            f.write(f"  Dropout率: {DROPOUT_RATE}\n")
            f.write(f"  总参数数: {sum(p.numel() for p in model.parameters()):,}\n")
        print(f"模型结构已保存到: {Path(MODEL_SUMMARY_PATH).resolve()}")
    except Exception as e:
        print(f"保存模型结构失败: {e}")

    # 7. 训练模型
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=MAX_EPOCHS)

    # 7.1 保存训练历史
    try:
        history_df = pd.DataFrame({'epoch': np.arange(1, len(train_losses)+1), 'train_loss': train_losses, 'val_loss': val_losses})
        history_df.to_csv(Path(TRAINING_HISTORY_PATH).resolve(), index=False)
        print(f"训练历史已保存到: {Path(TRAINING_HISTORY_PATH).resolve()}")
    except Exception as e:
        print(f"保存训练历史失败: {e}")

    # 8. 在独立测试集上评估模型
    print("\n=== 在独立测试集上评估模型 ===")
    test_predictions, test_actuals = evaluate_on_test_set(model, test_loader, scaler_y)

    # 8.5 计算训练集预测并反归一化（用于保存指标）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    train_predictions = []
    train_actuals = []
    with torch.no_grad():
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            train_predictions.extend(outputs.cpu().numpy())
            train_actuals.extend(y_batch.numpy())
    if len(train_predictions) > 0:
        train_predictions = scaler_y.inverse_transform(np.array(train_predictions).reshape(-1, len(RIGIDITY_VALUES)))
        train_actuals = scaler_y.inverse_transform(np.array(train_actuals).reshape(-1, len(RIGIDITY_VALUES)))

    # 保存主要性能指标到txt文件
    try:
        save_metrics_to_txt(train_actuals, train_predictions, test_actuals, test_predictions, filename=MODEL_METRICS_PATH)
        print(f"模型主要性能指标已自动保存到: {Path(MODEL_METRICS_PATH).resolve()}")
    except Exception as e:
        print(f"保存性能指标失败: {e}")

    # 9. 扩展预测（动态至太阳数据末日）
    print("\n=== 开始扩展预测 ===")
    start_date = pd.to_datetime(cosmic_data['date YYYY-MM-DD'].min())
    end_date = pd.to_datetime(solar_data['date'].max())
    prediction_dates = create_prediction_dates(start_date, end_date)

    print(f"将预测从 {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}, 总共 {len(prediction_dates)} 个日期")

    extended_pred_dates, extended_predictions = predict_cosmic_ray_extended(
        model, solar_data, cosmic_data, prediction_dates, scaler_X, scaler_y, sequence_length=SEQUENCE_LENGTH
    )

    # 10. 计算详细指标并保存
    calculate_metrics_and_stats(cosmic_data, test_actuals, test_predictions, extended_pred_dates, extended_predictions, scaler_y)

    # 11. 保存完整结果
    try:
        save_complete_results(dates_test, test_actuals, test_predictions, extended_pred_dates, extended_predictions)
    except Exception as e:
        print(f"保存完整结果失败: {e}")

    # 12. 保存模型与归一化器
    try:
        torch.save({'model_state_dict': model.state_dict(), 'scaler_X': scaler_X, 'scaler_y': scaler_y,
                    'model_config': {'input_size': input_size, 'hidden_size': HIDDEN_SIZE, 'num_layers': NUM_LAYERS, 'output_size': output_size, 'sequence_length': SEQUENCE_LENGTH}},
                   str(Path(MODEL_SAVE_PATH).resolve()))
        print(f"模型已保存为: {Path(MODEL_SAVE_PATH).resolve()}")
    except Exception as e:
        print(f"保存模型失败: {e}")

    print("\n=== 完整LSTM模型训练和预测完成！===")


if __name__ == "__main__":
    main()