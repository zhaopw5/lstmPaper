import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import r2_score
from data_processor import load_and_check_data, RIGIDITY_VALUES


def plot_comprehensive_results(cosmic_data, train_losses, val_losses, test_predictions, test_actuals, test_dates, 
                              extended_pred_dates=None, extended_predictions=None, series_end_date=None):
    # 统一字体大小与加粗设置
    title_fs = 16
    label_fs = 14
    tick_fs = 12
    legend_fs = 12
    text_fs = 16
    fw = 'bold'

    print("Plotting comprehensive results...")
    
    # Create 2x2 layout for 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot 1: Training loss
    axes[0, 0].plot(train_losses, label='Training Loss', color='blue', alpha=0.8)
    axes[0, 0].plot(val_losses, label='Validation Loss', color='red', alpha=0.8)
    axes[0, 0].set_xlabel('Epoch', fontsize=label_fs, fontweight=fw)
    axes[0, 0].set_ylabel('Loss', fontsize=label_fs, fontweight=fw)
    axes[0, 0].set_title('Training Process', fontsize=title_fs, fontweight=fw)
    axes[0, 0].legend(prop={'size': legend_fs, 'weight': fw})
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='both', labelsize=tick_fs)
    
    # Subplot 2: Test prediction scatter (平均所有刚度)
    avg_test_actuals = np.mean(test_actuals, axis=1)
    avg_test_predictions = np.mean(test_predictions, axis=1)
    axes[0, 1].scatter(avg_test_actuals, avg_test_predictions, alpha=0.6, color='green')
    axes[0, 1].plot([avg_test_actuals.min(), avg_test_actuals.max()], [avg_test_actuals.min(), avg_test_actuals.max()], 'r--', lw=2)
    r2 = r2_score(avg_test_actuals, avg_test_predictions)
    axes[0, 1].set_xlabel('Actual Values (Avg)', fontsize=label_fs, fontweight=fw)
    axes[0, 1].set_ylabel('Predicted Values (Avg)', fontsize=label_fs, fontweight=fw)
    axes[0, 1].set_title(f'Test Set Prediction (R²={r2:.4f})', fontsize=title_fs, fontweight=fw)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='both', labelsize=tick_fs)
    
    # Subplot 3: Test time series (平均所有刚度)
    axes[1, 0].plot(test_dates, avg_test_actuals, label='Actual (Avg)', alpha=0.8, linewidth=2, color='blue')
    axes[1, 0].plot(test_dates, avg_test_predictions, label='Predicted (Avg)', alpha=0.8, linewidth=2, color='red')
    axes[1, 0].set_xlabel('Date', fontsize=label_fs, fontweight=fw)
    axes[1, 0].set_ylabel('Helium Flux (Avg)', fontsize=label_fs, fontweight=fw)
    axes[1, 0].set_title('Test Set Time Series', fontsize=title_fs, fontweight=fw)
    axes[1, 0].legend(prop={'size': legend_fs, 'weight': fw})
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='both', labelsize=tick_fs)
    axes[1, 0].tick_params(axis='x', rotation=30)
    
    # Subplot 4: Complete time series (平均所有刚度) (dynamic end by solar data)
    if extended_pred_dates is not None and extended_predictions is not None:
        obs_dates = cosmic_data['date YYYY-MM-DD']
        # 计算所有刚度的平均值
        rigidity_cols = [f'helium_{rigidity}GV' for rigidity in RIGIDITY_VALUES if f'helium_{rigidity}GV' in cosmic_data.columns]
        obs_flux = cosmic_data[rigidity_cols].mean(axis=1)
        avg_extended_predictions = np.mean(extended_predictions, axis=1)
        
        start_year = pd.to_datetime(obs_dates.iloc[0]).year
        end_year = pd.to_datetime(series_end_date if series_end_date is not None else obs_dates.iloc[-1]).year
        
        axes[1, 1].plot(obs_dates, obs_flux, 'b-', label='Observed Data (Avg)', linewidth=1.5, alpha=0.8)
        axes[1, 1].plot(extended_pred_dates, avg_extended_predictions, 'r-', label='LSTM Predictions (Avg)', linewidth=1.5, alpha=0.8)
        axes[1, 1].axvline(x=obs_dates.iloc[-1], color='green', linestyle='--', alpha=0.7, label='Observation End')
        axes[1, 1].set_title(f'Complete Time Series ({start_year}-{end_year})', fontsize=title_fs, fontweight=fw)
        axes[1, 1].set_xlabel('Date', fontsize=label_fs, fontweight=fw)
        axes[1, 1].set_ylabel('Helium Flux (Avg) (m⁻²sr⁻¹s⁻¹GV⁻¹)', fontsize=label_fs, fontweight=fw)
        axes[1, 1].legend(prop={'size': legend_fs, 'weight': fw})
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        axes[1, 1].xaxis.set_major_locator(mdates.YearLocator())
        axes[1, 1].tick_params(axis='both', labelsize=tick_fs)
        axes[1, 1].tick_params(axis='x', rotation=30)
    else:
        # If no extended predictions, show model performance metrics
        residuals = avg_test_actuals - avg_test_predictions
        mse = np.mean(residuals**2)
        mae = np.mean(np.abs(residuals))
        r2 = r2_score(avg_test_actuals, avg_test_predictions)
        
        axes[1, 1].text(0.1, 0.8, f'MSE (Avg): {mse:.6f}', transform=axes[1, 1].transAxes, fontsize=text_fs, fontweight=fw)
        axes[1, 1].text(0.1, 0.6, f'MAE (Avg): {mae:.6f}', transform=axes[1, 1].transAxes, fontsize=text_fs, fontweight=fw)
        axes[1, 1].text(0.1, 0.4, f'R² (Avg): {r2:.6f}', transform=axes[1, 1].transAxes, fontsize=text_fs, fontweight=fw)
        axes[1, 1].text(0.1, 0.2, f'Samples: {len(avg_test_actuals)}', transform=axes[1, 1].transAxes, fontsize=text_fs, fontweight=fw)
        axes[1, 1].set_title('Model Performance Metrics', fontsize=title_fs, fontweight=fw)
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    out_img = Path('LSTM_Prediction_Results_Overview.png').resolve()
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    print(f"总览图像已保存到: {out_img}")


def plot_individual_rigidity_results(cosmic_data, test_predictions, test_actuals, test_dates, 
                                    extended_pred_dates=None, extended_predictions=None, series_end_date=None):
    """为每个刚度生成单独的预测图"""
    print("Plotting individual rigidity results...")
    
    title_fs = 16
    label_fs = 14
    tick_fs = 12
    legend_fs = 12
    fw = 'bold'
    
    for i, rigidity in enumerate(RIGIDITY_VALUES):
        rigidity_name = f'{rigidity}GV'
        rigidity_col = f'helium_{rigidity}GV'
        
        if rigidity_col not in cosmic_data.columns:
            print(f"警告: 刚度 {rigidity_name} 在数据中不存在，跳过")
            continue
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'LSTM Prediction Results - Rigidity {rigidity_name}', fontsize=title_fs, fontweight=fw)
        
        # 获取当前刚度的数据
        test_actual_i = test_actuals[:, i] if test_actuals.ndim > 1 else test_actuals
        test_pred_i = test_predictions[:, i] if test_predictions.ndim > 1 else test_predictions
        
        # Subplot 1: 散点图
        axes[0, 0].scatter(test_actual_i, test_pred_i, alpha=0.6, color='green')
        axes[0, 0].plot([test_actual_i.min(), test_actual_i.max()], [test_actual_i.min(), test_actual_i.max()], 'r--', lw=2)
        r2 = r2_score(test_actual_i, test_pred_i)
        axes[0, 0].set_xlabel('Actual Values', fontsize=label_fs, fontweight=fw)
        axes[0, 0].set_ylabel('Predicted Values', fontsize=label_fs, fontweight=fw)
        axes[0, 0].set_title(f'Test Set Prediction (R²={r2:.4f})', fontsize=title_fs, fontweight=fw)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='both', labelsize=tick_fs)
        
        # Subplot 2: 测试集时序图
        axes[0, 1].plot(test_dates, test_actual_i, label='Actual', alpha=0.8, linewidth=2, color='blue')
        axes[0, 1].plot(test_dates, test_pred_i, label='Predicted', alpha=0.8, linewidth=2, color='red')
        axes[0, 1].set_xlabel('Date', fontsize=label_fs, fontweight=fw)
        axes[0, 1].set_ylabel('Helium Flux', fontsize=label_fs, fontweight=fw)
        axes[0, 1].set_title('Test Set Time Series', fontsize=title_fs, fontweight=fw)
        axes[0, 1].legend(prop={'size': legend_fs, 'weight': fw})
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='both', labelsize=tick_fs)
        axes[0, 1].tick_params(axis='x', rotation=30)
        
        # Subplot 3: 完整时序图
        if extended_pred_dates is not None and extended_predictions is not None:
            obs_dates = cosmic_data['date YYYY-MM-DD']
            obs_flux = cosmic_data[rigidity_col]
            extended_pred_i = extended_predictions[:, i]
            
            start_year = pd.to_datetime(obs_dates.iloc[0]).year
            end_year = pd.to_datetime(series_end_date if series_end_date is not None else obs_dates.iloc[-1]).year
            
            axes[1, 0].plot(obs_dates, obs_flux, 'b-', label='Observed Data', linewidth=1.5, alpha=0.8)
            axes[1, 0].plot(extended_pred_dates, extended_pred_i, 'r-', label='LSTM Predictions', linewidth=1.5, alpha=0.8)
            axes[1, 0].axvline(x=obs_dates.iloc[-1], color='green', linestyle='--', alpha=0.7, label='Observation End')
            axes[1, 0].set_title(f'Complete Time Series ({start_year}-{end_year})', fontsize=title_fs, fontweight=fw)
            axes[1, 0].set_xlabel('Date', fontsize=label_fs, fontweight=fw)
            axes[1, 0].set_ylabel('Helium Flux (m⁻²sr⁻¹s⁻¹GV⁻¹)', fontsize=label_fs, fontweight=fw)
            axes[1, 0].legend(prop={'size': legend_fs, 'weight': fw})
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            axes[1, 0].xaxis.set_major_locator(mdates.YearLocator())
            axes[1, 0].tick_params(axis='both', labelsize=tick_fs)
            axes[1, 0].tick_params(axis='x', rotation=30)
        else:
            axes[1, 0].text(0.5, 0.5, 'No Extended Predictions Available', 
                           transform=axes[1, 0].transAxes, fontsize=label_fs, 
                           ha='center', va='center', fontweight=fw)
            axes[1, 0].axis('off')
        
        # Subplot 4: 性能指标
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        residuals = test_actual_i - test_pred_i
        mse = mean_squared_error(test_actual_i, test_pred_i)
        mae = mean_absolute_error(test_actual_i, test_pred_i)
        mape = np.mean(np.abs(residuals / test_actual_i)) * 100
        
        axes[1, 1].text(0.1, 0.8, f'MSE: {mse:.6f}', transform=axes[1, 1].transAxes, fontsize=label_fs, fontweight=fw)
        axes[1, 1].text(0.1, 0.6, f'MAE: {mae:.6f}', transform=axes[1, 1].transAxes, fontsize=label_fs, fontweight=fw)
        axes[1, 1].text(0.1, 0.4, f'R²: {r2:.6f}', transform=axes[1, 1].transAxes, fontsize=label_fs, fontweight=fw)
        axes[1, 1].text(0.1, 0.2, f'MAPE: {mape:.2f}%', transform=axes[1, 1].transAxes, fontsize=label_fs, fontweight=fw)
        axes[1, 1].set_title('Performance Metrics', fontsize=title_fs, fontweight=fw)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        out_img = Path(f'LSTM_Prediction_Results_{rigidity_name}.png').resolve()
        plt.savefig(out_img, dpi=300, bbox_inches='tight')
        print(f"刚度 {rigidity_name} 图像已保存到: {out_img}")
        plt.close(fig)  # 关闭图像释放内存


def main():
    # 读取训练历史
    hist_path = Path('training_history.csv')
    print(f"读取训练历史: {hist_path.resolve()}")
    history = pd.read_csv(hist_path)
    train_losses = history['train_loss'].values
    val_losses = history['val_loss'].values

    # 读取测试集预测 - 现在是多刚度格式
    test_path = Path('test_set_predictions.csv')
    print(f"读取测试集预测: {test_path.resolve()}")
    test_df = pd.read_csv(test_path, parse_dates=['date'])
    
    # 重构多刚度数据为矩阵格式
    unique_dates = test_df['date'].unique()
    unique_rigidities = test_df['rigidity'].unique()
    
    test_dates = unique_dates
    test_actuals = np.zeros((len(unique_dates), len(unique_rigidities)))
    test_predictions = np.zeros((len(unique_dates), len(unique_rigidities)))
    
    for i, date in enumerate(unique_dates):
        for j, rigidity in enumerate(unique_rigidities):
            mask = (test_df['date'] == date) & (test_df['rigidity'] == rigidity)
            if mask.sum() > 0:
                test_actuals[i, j] = test_df[mask]['actual_flux'].iloc[0]
                test_predictions[i, j] = test_df[mask]['predicted_flux'].iloc[0]

    # 读取观测与太阳数据（用于完整时序对比与动态末日）
    solar_data, cosmic_data = load_and_check_data()
    solar_end_date = solar_data['date'].max()

    # 读取（可选）扩展预测（并截断到太阳数据末日）- 现在是多刚度格式
    extended_pred_dates = None
    extended_predictions = None
    ext_path = Path('cosmic_ray_predictions_extended.csv')
    if ext_path.exists():
        print(f"读取扩展预测: {ext_path.resolve()}")
        ext_df = pd.read_csv(ext_path, parse_dates=['date'])
        ext_df = ext_df[ext_df['date'] <= solar_end_date]
        if not ext_df.empty:
            # 重构扩展预测数据为矩阵格式
            unique_ext_dates = ext_df['date'].unique()
            unique_ext_rigidities = ext_df['rigidity'].unique()
            
            extended_pred_dates = unique_ext_dates
            extended_predictions = np.zeros((len(unique_ext_dates), len(unique_ext_rigidities)))
            
            for i, date in enumerate(unique_ext_dates):
                for j, rigidity in enumerate(unique_ext_rigidities):
                    mask = (ext_df['date'] == date) & (ext_df['rigidity'] == rigidity)
                    if mask.sum() > 0:
                        extended_predictions[i, j] = ext_df[mask]['predicted_flux'].iloc[0]
            
            print(f"扩展预测已截断至太阳数据最后一天: {solar_end_date.date()}")
        else:
            print("扩展预测在太阳数据时间范围内为空，跳过完整时序对比。")
    else:
        print("未发现扩展预测CSV，跳过完整时序对比。")

    # 绘制总览图
    plot_comprehensive_results(
        cosmic_data=cosmic_data,
        train_losses=train_losses,
        val_losses=val_losses,
        test_predictions=test_predictions,
        test_actuals=test_actuals,
        test_dates=test_dates,
        extended_pred_dates=extended_pred_dates,
        extended_predictions=extended_predictions,
        series_end_date=solar_end_date
    )
    
    # 绘制每个刚度的单独图表
    plot_individual_rigidity_results(
        cosmic_data=cosmic_data,
        test_predictions=test_predictions,
        test_actuals=test_actuals,
        test_dates=test_dates,
        extended_pred_dates=extended_pred_dates,
        extended_predictions=extended_predictions,
        series_end_date=solar_end_date
    )

if __name__ == "__main__":
    main()