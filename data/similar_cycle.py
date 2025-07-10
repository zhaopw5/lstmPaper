import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import signal
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# 设置字体显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

class ImprovedSolarCycleSimilarityPredictor:
    """
    **改进版太阳周期预测器**
    
    **特点：**
    - 太阳黑子等参数：基于11年周期
    - 太阳极性：基于22年海尔周期独立预测
    """
    
    def __init__(self):
        """**初始化预测器**"""
        self.SOLAR_PARAMETERS = ['HMF', 'wind_speed', 'SSN', 'polarity', 'HCS_tilt']
        self.SOLAR_CYCLE_YEARS = 11      # **太阳黑子周期**
        self.POLARITY_CYCLE_YEARS = 22   # **太阳极性周期（海尔周期）**
        
        # **参数权重（极性单独处理）**
        self.PARAMETER_WEIGHTS = {
            'SSN': 0.6,
            'HMF': 0.15,
            'wind_speed': 0.1,
            'HCS_tilt': 0.15
            # **polarity不参与相似度计算，单独预测**
        }
        
    def load_solar_data(self, file_path):
        """**加载太阳数据**"""
        print("🌞 **正在加载太阳数据...**")
        
        solar_data = pd.read_csv(file_path)
        solar_data['date'] = pd.to_datetime(solar_data['date'])
        solar_data = solar_data.sort_values('date')
        
        print(f"📅 **数据时间范围:** {solar_data['date'].min()} 到 {solar_data['date'].max()}")
        print(f"📊 **总天数:** {len(solar_data)} 天")
        print(f"⏰ **时间覆盖:** {(solar_data['date'].max() - solar_data['date'].min()).days / 365.25:.1f} 年")
        
        return solar_data
        
    def smooth_data(self, data, window_size=397):
        """**数据平滑处理**"""
        if len(data) < window_size:
            return data
            
        smoothed_data = pd.Series(data).rolling(
            window=window_size, 
            center=True, 
            min_periods=window_size//3
        ).mean()
        
        smoothed_data = smoothed_data.fillna(method='bfill').fillna(method='ffill')
        return smoothed_data.values
    
    def identify_polarity_reversals(self, solar_data):
        """
        **识别太阳极性反转事件**
        
        **太阳极性特点：**
        - 每个太阳黑子周期结束时发生极性反转
        - 完整的极性周期约22年
        - 极性值通常为+1或-1
        """
        print("🧲 **正在识别太阳极性反转事件...**")
        
        polarity_data = solar_data['polarity'].values
        date_data = solar_data['date'].values
        
        # **平滑极性数据以识别长期趋势**
        polarity_smoothed = self.smooth_data(polarity_data, window_size=365)  # **1年平滑**
        
        # **寻找极性反转点（符号变化）**
        reversals = []
        for i in range(1, len(polarity_smoothed)):
            if polarity_smoothed[i] * polarity_smoothed[i-1] < 0:  # **符号变化**
                reversals.append(i)
        
        print(f"🔄 **找到{len(reversals)}个极性反转事件:**")
        
        # **构建极性周期**
        polarity_cycles = []
        for i in range(len(reversals) - 1):
            start_idx = reversals[i]
            end_idx = reversals[i + 1]
            
            cycle_length_days = end_idx - start_idx
            cycle_length_years = cycle_length_days / 365.25
            
            # **极性半周期通常10-12年**
            if 8 <= cycle_length_years <= 15:
                cycle_data = solar_data.iloc[start_idx:end_idx].copy().reset_index(drop=True)
                cycle_data['polarity_phase'] = np.linspace(0, 1, len(cycle_data))
                
                polarity_info = {
                    'cycle_id': len(polarity_cycles) + 1,
                    'start_date': pd.to_datetime(date_data[start_idx]),
                    'end_date': pd.to_datetime(date_data[end_idx]),
                    'length_years': cycle_length_years,
                    'start_polarity': polarity_smoothed[start_idx],
                    'end_polarity': polarity_smoothed[end_idx],
                    'data': cycle_data
                }
                
                polarity_cycles.append(polarity_info)
                
                print(f"   **极性周期{polarity_info['cycle_id']}:** "
                      f"{polarity_info['start_date'].strftime('%Y-%m-%d')} → "
                      f"{polarity_info['end_date'].strftime('%Y-%m-%d')} "
                      f"({polarity_info['length_years']:.1f}年)")
        
        return polarity_cycles, polarity_smoothed
    
    def predict_polarity_separately(self, solar_data, prediction_start_date, prediction_days=365):
        """
        **单独预测太阳极性**
        
        **方法：**
        1. 识别历史极性反转模式
        2. 基于22年海尔周期预测
        3. 考虑极性的渐变特性
        """
        print("🧲 **开始单独预测太阳极性...**")
        
        # **获取历史极性数据**
        historical_data = solar_data[solar_data['date'] < prediction_start_date]
        polarity_cycles, polarity_smoothed = self.identify_polarity_reversals(historical_data)
        
        if len(polarity_cycles) == 0:
            print("⚠️ **未找到足够的极性反转事件，使用简化预测**")
            return [-1] * prediction_days  # **默认极性**
        
        # **分析当前极性状态**
        current_polarity = historical_data['polarity'].iloc[-100:].median()  # **近期极性**
        last_reversal_date = polarity_cycles[-1]['end_date']
        
        # **计算距离上次反转的时间**
        days_since_reversal = (prediction_start_date - last_reversal_date).days
        avg_half_cycle_length = np.mean([cycle['length_years'] for cycle in polarity_cycles]) * 365.25
        
        print(f"📊 **极性预测参数:**")
        print(f"   **当前极性:** {current_polarity:.1f}")
        print(f"   **上次反转:** {last_reversal_date.strftime('%Y-%m-%d')}")
        print(f"   **距离上次反转:** {days_since_reversal}天")
        print(f"   **平均半周期长度:** {avg_half_cycle_length:.0f}天")
        
        # **预测极性序列**
        polarity_predictions = []
        
        for day_offset in range(prediction_days):
            total_days_since_reversal = days_since_reversal + day_offset
            
            # **判断是否可能发生反转**
            if total_days_since_reversal > avg_half_cycle_length * 0.8:
                # **接近反转期，极性可能开始变化**
                reversal_probability = min(1.0, (total_days_since_reversal - avg_half_cycle_length * 0.8) / 
                                         (avg_half_cycle_length * 0.4))
                
                if reversal_probability > 0.5:
                    # **发生反转**
                    predicted_polarity = -current_polarity
                else:
                    # **渐变过程**
                    predicted_polarity = current_polarity * (1 - reversal_probability)
            else:
                # **稳定期，保持当前极性**
                predicted_polarity = current_polarity
            
            # **确保极性值在合理范围内**
            if abs(predicted_polarity) < 0.3:
                predicted_polarity = 0  # **接近零的极性**
            elif predicted_polarity > 0:
                predicted_polarity = 1
            else:
                predicted_polarity = -1
            
            polarity_predictions.append(predicted_polarity)
        
        print(f"🎯 **极性预测完成：预测了{len(polarity_predictions)}天的极性变化**")
        
        return polarity_predictions
    
    def identify_complete_solar_cycles(self, solar_data, primary_param='SSN'):
        """**识别完整太阳周期（用于非极性参数）**"""
        print(f"🔍 **识别完整太阳周期（用于{primary_param}等参数）...**")
        
        ssn_data = solar_data[primary_param].values
        date_data = solar_data['date'].values
        ssn_smoothed = self.smooth_data(ssn_data, window_size=397)
        
        ssn_mean = np.nanmean(ssn_smoothed)
        ssn_std = np.nanstd(ssn_smoothed)
        
        print(f"📈 **SSN统计信息:** 平均值={ssn_mean:.1f}, 标准差={ssn_std:.1f}")
        
        # **寻找极小期**
        min_distance = int(6 * 365.25)
        valleys, _ = signal.find_peaks(
            -ssn_smoothed,
            height=-(ssn_mean - 0.3 * ssn_std),
            distance=min_distance,
            prominence=ssn_std * 0.3
        )
        
        if len(valleys) < 3:
            valleys, _ = signal.find_peaks(
                -ssn_smoothed,
                height=-(ssn_mean + 0.1 * ssn_std),
                distance=int(4 * 365.25),
                prominence=ssn_std * 0.1
            )
        
        print(f"🌑 **找到{len(valleys)}个太阳活动极小期**")
        
        # **构建完整周期**
        complete_cycles = []
        for i in range(len(valleys) - 1):
            start_idx = valleys[i]
            end_idx = valleys[i + 1]
            
            cycle_length_days = end_idx - start_idx
            cycle_length_years = cycle_length_days / 365.25
            
            if 8 <= cycle_length_years <= 15:
                cycle_data = solar_data.iloc[start_idx:end_idx+1].copy().reset_index(drop=True)
                cycle_data['cycle_phase'] = np.linspace(0, 1, len(cycle_data))
                
                cycle_ssn = ssn_smoothed[start_idx:end_idx+1]
                max_idx_in_cycle = np.argmax(cycle_ssn)
                
                cycle_info = {
                    'cycle_id': len(complete_cycles) + 1,
                    'start_date': pd.to_datetime(date_data[start_idx]),
                    'end_date': pd.to_datetime(date_data[end_idx]),
                    'length_days': cycle_length_days,
                    'length_years': cycle_length_years,
                    'data': cycle_data,
                    'ssn_smoothed': cycle_ssn,
                    'max_ssn': cycle_ssn[max_idx_in_cycle],
                    'min_ssn': min(cycle_ssn[0], cycle_ssn[-1]),
                    'ssn_amplitude': cycle_ssn[max_idx_in_cycle] - min(cycle_ssn[0], cycle_ssn[-1])
                }
                
                complete_cycles.append(cycle_info)
        
        print(f"🔄 **识别了{len(complete_cycles)}个完整太阳周期**")
        
        return complete_cycles, ssn_smoothed
    
    def calculate_similarity_without_polarity(self, cycle1, cycle2):
        """**计算相似度（不包含极性）**"""
        similarity_scores = {}
        total_similarity = 0
        total_weight = 0
        
        # **排除极性的参数列表**
        params_without_polarity = [p for p in self.SOLAR_PARAMETERS if p != 'polarity']
        
        for param in params_without_polarity:
            if param in cycle1['data'].columns and param in cycle2['data'].columns:
                try:
                    phase1 = cycle1['data']['cycle_phase'].values
                    phase2 = cycle2['data']['cycle_phase'].values
                    
                    values1 = cycle1['data'][param].values
                    values2 = cycle2['data'][param].values
                    
                    if len(phase1) > 10 and len(phase2) > 10:
                        unified_phase = np.linspace(0, 1, 200)
                        
                        valid1 = ~np.isnan(values1)
                        valid2 = ~np.isnan(values2)
                        
                        if valid1.sum() > 5 and valid2.sum() > 5:
                            interp_func1 = interp1d(phase1[valid1], values1[valid1], 
                                                   kind='linear', bounds_error=False, 
                                                   fill_value='extrapolate')
                            interp_func2 = interp1d(phase2[valid2], values2[valid2], 
                                                   kind='linear', bounds_error=False, 
                                                   fill_value='extrapolate')
                            
                            values1_interp = interp_func1(unified_phase)
                            values2_interp = interp_func2(unified_phase)
                            
                            correlation = np.corrcoef(values1_interp, values2_interp)[0, 1]
                            similarity = abs(correlation) if not np.isnan(correlation) else 0
                            
                            similarity_scores[param] = similarity
                            weight = self.PARAMETER_WEIGHTS.get(param, 0.1)
                            total_similarity += similarity * weight
                            total_weight += weight
                        else:
                            similarity_scores[param] = 0
                    else:
                        similarity_scores[param] = 0
                except:
                    similarity_scores[param] = 0
            else:
                similarity_scores[param] = 0
        
        final_similarity = total_similarity / total_weight if total_weight > 0 else 0
        return final_similarity, similarity_scores
    
    def predict_solar_parameters_with_separate_polarity(self, solar_data, prediction_start_date, prediction_days=365):
        """
        **综合预测：其他参数基于22年周期，极性单独预测**
        """
        print("🔮 **开始综合预测（极性单独处理）...**")
        
        # **1. 单独预测极性**
        polarity_predictions = self.predict_polarity_separately(
            solar_data, prediction_start_date, prediction_days)
        
        # **2. 基于11年周期预测其他参数**
        historical_data = solar_data[solar_data['date'] < prediction_start_date]
        complete_cycles, smoothed_ssn = self.identify_complete_solar_cycles(historical_data)
        
        if len(complete_cycles) == 0:
            print("❌ **无法识别完整的太阳周期**")
            return None
        
        # **找相似周期**
        last_cycle = complete_cycles[-1]
        similarity_results = []
        
        for cycle in complete_cycles[:-1]:  # **排除最后一个周期**
            total_similarity, _ = self.calculate_similarity_without_polarity(last_cycle, cycle)
            similarity_results.append({
                'cycle': cycle,
                'similarity': total_similarity
            })
        
        similarity_results.sort(key=lambda x: x['similarity'], reverse=True)
        similar_cycles = similarity_results[:min(3, len(similarity_results))]
        
        print(f"🎯 **找到{len(similar_cycles)}个相似周期用于预测非极性参数**")
        
        # **3. 预测其他参数**
        prediction_dates = pd.date_range(start=prediction_start_date, periods=prediction_days, freq='D')
        prediction_results = {param: [] for param in self.SOLAR_PARAMETERS}
        
        # **计算当前相位**
        days_since_last_cycle = (prediction_start_date - last_cycle['end_date']).days
        avg_cycle_length = np.mean([cycle['length_days'] for cycle in complete_cycles])
        current_phase_start = days_since_last_cycle / avg_cycle_length
        
        print(f"📊 **预测配置:**")
        print(f"   **当前相位:** {current_phase_start:.3f}")
        print(f"   **平均周期长度:** {avg_cycle_length:.0f}天")
        
        # **预测非极性参数**
        non_polarity_params = [p for p in self.SOLAR_PARAMETERS if p != 'polarity']
        
        for day_offset in range(prediction_days):
            current_phase = (current_phase_start + day_offset / avg_cycle_length) % 1.0
            
            for param in non_polarity_params:
                param_predictions = []
                weights = []
                
                for similar_cycle in similar_cycles:
                    cycle_data = similar_cycle['cycle']['data']
                    similarity_score = similar_cycle['similarity']
                    
                    if param in cycle_data.columns:
                        try:
                            cycle_phases = cycle_data['cycle_phase'].values
                            param_values = cycle_data[param].values
                            
                            valid_idx = ~np.isnan(param_values)
                            if valid_idx.sum() > 5:
                                valid_phases = cycle_phases[valid_idx]
                                valid_values = param_values[valid_idx]
                                
                                if len(valid_phases) > 1:
                                    interp_func = interp1d(valid_phases, valid_values, 
                                                         kind='linear', bounds_error=False, 
                                                         fill_value='extrapolate')
                                    predicted_value = interp_func(current_phase)
                                    
                                    if not np.isnan(predicted_value):
                                        param_predictions.append(predicted_value)
                                        weights.append(similarity_score)
                        except:
                            continue
                
                # **计算加权平均**
                if param_predictions and weights:
                    weight_array = np.array(weights)
                    weight_array = weight_array / weight_array.sum()
                    final_prediction = np.average(param_predictions, weights=weight_array)
                    
                    # **参数约束**
                    if param == 'SSN':
                        final_prediction = max(0, final_prediction)
                    elif param == 'wind_speed':
                        final_prediction = np.clip(final_prediction, 200, 800)
                    elif param == 'HMF':
                        final_prediction = max(0.1, final_prediction)
                    elif param == 'HCS_tilt':
                        final_prediction = np.clip(final_prediction, 0, 90)
                else:
                    # **默认值**
                    default_values = {
                        'HMF': 5.0, 'wind_speed': 400.0, 'SSN': 50.0, 'HCS_tilt': 30.0
                    }
                    final_prediction = default_values.get(param, 0)
                
                prediction_results[param].append(final_prediction)
        
        # **4. 添加极性预测结果**
        prediction_results['polarity'] = polarity_predictions
        
        # **5. 创建最终结果**
        prediction_df = pd.DataFrame(prediction_results)
        prediction_df['date'] = prediction_dates
        
        print(f"\n📈 **预测完成统计:**")
        for param in self.SOLAR_PARAMETERS:
            if param == 'polarity':
                unique_vals = np.unique(prediction_df[param])
                print(f"   **{param}:** 预测值 = {unique_vals}")
            else:
                min_val = prediction_df[param].min()
                max_val = prediction_df[param].max()
                mean_val = prediction_df[param].mean()
                print(f"   **{param}:** 范围=[{min_val:.2f}, {max_val:.2f}], 平均={mean_val:.2f}")
        
        return prediction_df, complete_cycles, smoothed_ssn
    
    def visualize_results(self, solar_data, complete_cycles, prediction_results, 
                         prediction_start_date, smoothed_ssn):
        """**可视化结果**"""
        print("📊 **生成可视化图表...**")
        
        fig, axes = plt.subplots(3, 2, figsize=(24, 18))
        axes = axes.flatten()
        
        param_display_names = {
            'HMF': 'Heliospheric Magnetic Field (nT)',
            'wind_speed': 'Solar Wind Speed (km/s)',
            'SSN': 'Sunspot Number',
            'polarity': 'Magnetic Polarity (22-year cycle)',
            'HCS_tilt': 'HCS Tilt Angle (degrees)'
        }
        
        for i, param in enumerate(self.SOLAR_PARAMETERS):
            ax = axes[i]
            
            # **历史数据**
            historical_data = solar_data[solar_data['date'] < prediction_start_date]
            ax.plot(historical_data['date'], historical_data[param], 
                   label='Historical Data', color='blue', alpha=0.7, linewidth=2)
            
            # **SSN平滑曲线**
            if param == 'SSN' and smoothed_ssn is not None:
                ax.plot(historical_data['date'], smoothed_ssn[:len(historical_data)], 
                       label='Smoothed SSN', color='green', linewidth=3)
            
            # **标记周期**
            if param != 'polarity':
                # **对于非极性参数，显示11年周期**
                cycle_colors = ['red', 'orange', 'purple', 'brown']
                for j, cycle in enumerate(complete_cycles):
                    color = cycle_colors[j % len(cycle_colors)]
                    ax.axvspan(cycle['start_date'], cycle['end_date'], 
                              alpha=0.3, color=color, 
                              label=f'11-year Cycle {cycle["cycle_id"]}' if i == 0 else '')
            
            # **预测结果**
            if param == 'polarity':
                ax.plot(prediction_results['date'], prediction_results[param], 
                       label='Polarity Prediction (22-year cycle)', 
                       color='red', linewidth=4, linestyle='--')
            else:
                ax.plot(prediction_results['date'], prediction_results[param], 
                       label='Prediction (11-year cycle)', 
                       color='red', linewidth=4, linestyle='--')
            
            # **预测起点**
            ax.axvline(x=prediction_start_date, color='black', linestyle='-', 
                      alpha=0.9, linewidth=3, label='Prediction Start' if i == 0 else '')
            
            # **设置图表属性**
            ax.set_title(param_display_names[param], fontsize=16, fontweight='bold')
            ax.set_xlabel('Date', fontsize=14, fontweight='bold')
            ax.set_ylabel(param_display_names[param], fontsize=14, fontweight='bold')
            
            if i == 0:
                legend = ax.legend(loc='upper left', fontsize=10)
                for text in legend.get_texts():
                    text.set_weight('bold')
            
            ax.grid(True, alpha=0.4)
            ax.tick_params(axis='x', rotation=45, labelsize=12)
            ax.tick_params(axis='y', labelsize=12)
            
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
        
        if len(self.SOLAR_PARAMETERS) < 6:
            axes[5].axis('off')
        
        plt.tight_layout()
        plt.suptitle(f'Solar Parameter Prediction (Polarity: 22-year cycle, Others: 11-year cycle)\nPrediction: {prediction_start_date.strftime("%Y-%m-%d")} to 2030-12-31', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.savefig('/home/phil/Files/lstmPaper/data/outputs/cycle_analysis/complete_solar_prediction_with_separate_polarity.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ **可视化完成！**")

def main():
    """**主函数**"""
    print("🌞 **改进版太阳参数预测系统**")
    print("=" * 80)
    print("🧲 **特色：太阳极性基于22年海尔周期单独预测**")
    print("📊 **其他参数：基于11年太阳黑子周期预测**")
    print("=" * 80)
    
    predictor = ImprovedSolarCycleSimilarityPredictor()
    
    solar_data_file = '/home/phil/Files/lstmPaper/data/outputs/cycle_analysis/solar_physics_data_1985_2025.csv'
    solar_data = predictor.load_solar_data(solar_data_file)
    
    prediction_start_date = datetime(2019, 10, 30)
    prediction_end_date = datetime(2030, 12, 31)
    prediction_days = (prediction_end_date - prediction_start_date).days + 1
    print(f"\n🎯 **预测开始日期:** {prediction_start_date.strftime('%Y-%m-%d')}")
    print(f"🎯 **预测结束日期:** {prediction_end_date.strftime('%Y-%m-%d')}")
    print(f"📅 **预测天数:** {prediction_days}天")
    
    # **综合预测**
    prediction_results, complete_cycles, smoothed_ssn = predictor.predict_solar_parameters_with_separate_polarity(
        solar_data, prediction_start_date, prediction_days=prediction_days)
    
    if prediction_results is None:
        print("❌ **预测失败**")
        return
    
    # **合并历史数据和预测数据**
    historical_data = solar_data[solar_data['date'] < prediction_start_date].copy()
    
    # **创建完整的数据集（历史数据 + 预测数据）**
    complete_data = pd.concat([historical_data[['date'] + predictor.SOLAR_PARAMETERS], 
                              prediction_results[['date'] + predictor.SOLAR_PARAMETERS]], 
                             ignore_index=True)
    
    # **重新排列列顺序，将date放在第一列**
    column_order = ['date'] + predictor.SOLAR_PARAMETERS
    complete_data = complete_data[column_order]
    
    # **保存完整结果（历史数据 + 预测数据）**
    output_dir = '/home/phil/Files/lstmPaper/data/outputs/cycle_analysis/'
    output_file = output_dir + 'solar_physics_data_1985_2025_cycle_prediction.csv'
    complete_data.to_csv(output_file, index=False)
    print(f"\n💾 **完整数据（历史+预测）已保存到:** {output_file}")
    print(f"📊 **数据统计:** 历史数据{len(historical_data)}天 + 预测数据{len(prediction_results)}天 = 总计{len(complete_data)}天")
    
    # **可视化**
    predictor.visualize_results(solar_data, complete_cycles, prediction_results, 
                               prediction_start_date, smoothed_ssn)
    
    print(f"\n✅ **改进版预测完成！极性已基于22年周期单独预测**")
    print(f"🔮 **预测时间范围:** {prediction_start_date.strftime('%Y-%m-%d')} 到 {prediction_end_date.strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    main()