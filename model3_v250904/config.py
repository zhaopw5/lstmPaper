"""
LSTM宇宙线预测模型配置文件
集中管理所有可配置参数，避免硬编码魔法数字
修改这里的参数会影响整个项目
"""

# =============================================================================
# 数据处理配置
# =============================================================================
SEQUENCE_LENGTH = 180  # LSTM输入序列长度（天数）- 主要配置参数
PREDICTION_HORIZON = 1  # 预测未来天数

# 数据划分比例（按时间顺序：训练集->验证集->测试集）
TRAIN_RATIO = 0.80  # 前80%用于训练
VAL_RATIO = 0.1   # 中间10%用于验证（调参和早停）
TEST_RATIO = 0.1  # 后10%用于测试（模拟真实预测）

# =============================================================================
# 模型架构配置
# =============================================================================
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT_RATE = 0.05

# =============================================================================
# 训练配置
# =============================================================================
BATCH_SIZE = 64
MAX_EPOCHS = 1000
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0
PATIENCE = 30  # 早停耐心值
GRAD_CLIP_NORM = 1.0  # 梯度裁剪阈值

# =============================================================================
# 数据路径配置
# =============================================================================
SOLAR_DATA_PATH = "/home/phil/Files/lstmPaper/data/outputs/cycle_analysis/solar_physics_data_1985_2025_v250820.csv"
COSMIC_DATA_PATH = "/home/phil/Files/lstmPaper/data/raw_data/ams/helium.csv"

# =============================================================================
# 输出文件配置
# =============================================================================
MODEL_SAVE_PATH = "complete_lstm_model.pth"
BEST_MODEL_PATH = "best_corrected_model.pth"
TRAINING_HISTORY_PATH = "training_history.csv"
MODEL_SUMMARY_PATH = "model_summary.txt"
MODEL_METRICS_PATH = "model_metrics.txt"
TEST_PREDICTIONS_PATH = "test_set_predictions.csv"
EXTENDED_PREDICTIONS_PATH = "cosmic_ray_predictions_extended.csv"

# =============================================================================
# 说明信息
# =============================================================================
def print_config_info():
    """打印当前配置信息"""
    print("=== 当前配置参数 ===")
    print(f"序列长度: {SEQUENCE_LENGTH} 天")
    print(f"隐藏层大小: {HIDDEN_SIZE}")
    print(f"LSTM层数: {NUM_LAYERS}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"最大训练轮数: {MAX_EPOCHS}")
    print(f"学习率: {LEARNING_RATE}")
    print(f"早停耐心值: {PATIENCE}")
    print("=" * 25)