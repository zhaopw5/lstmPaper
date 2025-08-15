import pandas as pd
from pathlib import Path
from visualization import plot_comprehensive_results
from data_processor import load_and_check_data

def main():
    # 读取训练历史
    hist_path = Path('training_history.csv')
    print(f"读取训练历史: {hist_path.resolve()}")
    history = pd.read_csv(hist_path)
    train_losses = history['train_loss'].values
    val_losses = history['val_loss'].values

    # 读取测试集预测
    test_path = Path('测试集预测结果.csv')
    print(f"读取测试集预测: {test_path.resolve()}")
    test_df = pd.read_csv(test_path, parse_dates=['date'])
    test_dates = test_df['date'].values
    test_actuals = test_df['actual_flux'].values
    test_predictions = test_df['predicted_flux'].values

    # 读取（可选）扩展预测
    ext_path = Path('宇宙线预测结果_2011_2025.csv')
    extended_pred_dates = None
    extended_predictions = None
    if ext_path.exists():
        print(f"读取扩展预测: {ext_path.resolve()}")
        ext_df = pd.read_csv(ext_path, parse_dates=['date'])
        extended_pred_dates = ext_df['date'].values
        extended_predictions = ext_df['predicted_flux'].values
    else:
        print("未发现扩展预测CSV，跳过完整时序对比。")

    # 读取观测数据（用于完整时序对比）
    _, cosmic_data = load_and_check_data()

    # 出图
    plot_comprehensive_results(
        cosmic_data=cosmic_data,
        train_losses=train_losses,
        val_losses=val_losses,
        test_predictions=test_predictions,
        test_actuals=test_actuals,
        test_dates=test_dates,
        extended_pred_dates=extended_pred_dates,
        extended_predictions=extended_predictions
    )

if __name__ == "__main__":
    main()