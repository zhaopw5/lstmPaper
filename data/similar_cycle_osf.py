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
        self.SOLAR_PARAMETERS = ['HMF', 'wind_speed', 'HCS_tilt', 'polarity', 'SSN', 'daily_OSF']
        self.SOLAR_CYCLE_YEARS = 11      # **太阳黑子周期**
        self.POLARITY_CYCLE_YEARS = 22   # **太阳极性周期（海尔周期）**
        
        # **参数权重（极性和OSF单独处理）**
        self.PARAMETER_WEIGHTS = {
            'SSN': 0.6,         # **提高SSN权重，因为它是主要指标**
            'HMF': 0.2,         # **调整HMF权重**
            'wind_speed': 0.1,  # **保持风速权重**
            'HCS_tilt': 0.1     # **保持HCS倾斜角权重**
            # **polarity和daily_OSF不参与相似度计算，单独预测**
        }
        
        # —— 新增：可配置的随机种子与OSF噪声控制 ——
        self.random_seed = 12345         # 统一随机性
        self.osf_noise_mode = 'ar1'      # 'ar1' 或 'none'
        self.osf_vol_scale = 1.0         # 残差强度缩放，<1减小波动，>1放大
        
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
    
    def build_phase_templates(self, complete_cycles, params):
        """构建统一相位模板（对所有历史周期求均值）"""
        unified_phase = np.linspace(0, 1, 200)
        templates = {}
        for param in params:
            series = []
            for cycle in complete_cycles:
                if param in cycle['data'].columns:
                    phases = cycle['data']['cycle_phase'].values
                    values = cycle['data'][param].values
                    valid = ~np.isnan(values)
                    if valid.sum() > 5:
                        f = interp1d(phases[valid], values[valid], kind='linear', bounds_error=False, fill_value='extrapolate')
                        series.append(f(unified_phase))
            if series:
                templates[param] = np.nanmean(np.vstack(series), axis=0)
        return unified_phase, templates
    
    def predict_osf_from_ssn(self, historical_data, predicted_ssn):
        """
        **基于SSN预测OSF**
        
        **原理：**
        - OSF与SSN有强相关性，都反映太阳磁场活动
        - 通过历史数据建立SSN-OSF关系
        - 使用简单线性关系预测OSF
        """
        print("🌀 **基于SSN预测OSF...**")
        
        # 获取历史SSN和OSF数据
        historical_ssn = historical_data['SSN'].values
        historical_osf = historical_data['daily_OSF'].values
        
        # 打印原始数据统计
        print(f"📊 **原始数据统计:**")
        print(f"   **SSN范围:** [{np.nanmin(historical_ssn):.2f}, {np.nanmax(historical_ssn):.2f}]")
        print(f"   **OSF范围:** [{np.nanmin(historical_osf):.6f}, {np.nanmax(historical_osf):.6f}]")
        print(f"   **SSN NaN数量:** {np.isnan(historical_ssn).sum()}/{len(historical_ssn)}")
        print(f"   **OSF NaN数量:** {np.isnan(historical_osf).sum()}/{len(historical_osf)}")
        
        # 移除NaN值
        valid_idx = ~(np.isnan(historical_ssn) | np.isnan(historical_osf))
        clean_ssn = historical_ssn[valid_idx]
        clean_osf = historical_osf[valid_idx]
        
        print(f"   **有效数据点:** {len(clean_ssn)}/{len(historical_ssn)}")
        
        if len(clean_ssn) < 100:
            print("⚠️ **历史数据不足，使用简化OSF预测**")
            mean_historical_osf = np.nanmean(historical_osf)
            if np.isnan(mean_historical_osf):
                mean_historical_osf = 0.005  # 默认值
            return np.full(np.asarray(predicted_ssn).shape, mean_historical_osf, dtype=float)
        
        # 计算SSN-OSF相关性
        if len(clean_ssn) > 1 and len(clean_osf) > 1:
            correlation = np.corrcoef(clean_ssn, clean_osf)[0, 1]
        else:
            correlation = 0.0
        
        print(f"📊 **SSN-OSF相关系数:** {correlation:.3f}")
        
        # 计算线性回归参数
        mean_ssn = np.mean(clean_ssn)
        mean_osf = np.mean(clean_osf)
        std_ssn = np.std(clean_ssn)
        std_osf = np.std(clean_osf)
        
        print(f"📊 **统计参数:**")
        print(f"   **SSN:** 均值={mean_ssn:.2f}, 标准差={std_ssn:.2f}")
        print(f"   **OSF:** 均值={mean_osf:.6f}, 标准差={std_osf:.6f}")
        predicted_ssn = np.asarray(predicted_ssn, dtype=float)
        print(f"   **预测SSN范围:** [{predicted_ssn.min():.2f}, {predicted_ssn.max():.2f}]")
        
        # 使用标准化线性关系：OSF = slope * SSN + intercept
        if std_ssn > 0 and not np.isnan(correlation):
            slope = correlation * std_osf / std_ssn
            intercept = mean_osf - slope * mean_ssn
            
            # 预测OSF（线性部分）
            predicted_osf = slope * predicted_ssn + intercept
            
            print(f"✅ **线性回归参数:** slope={slope:.8f}, intercept={intercept:.6f}")
            print(f"   **预测前OSF范围:** [{predicted_osf.min():.6f}, {predicted_osf.max():.6f}]")

            # —— 基于历史残差加入AR(1)短期波动，并校准方差 ——
            try:
                if self.osf_noise_mode == 'none':
                    print("   **OSF噪声模式:** none（不注入残差）")
                else:
                    y_hat_hist = slope * clean_ssn + intercept
                    residuals = clean_osf - y_hat_hist
                    # AR(1)参数估计 φ
                    num = np.dot(residuals[1:], residuals[:-1])
                    den = np.dot(residuals[:-1], residuals[:-1]) + 1e-12
                    phi = np.clip(num / den, -0.99, 0.99)
                    eps = residuals[1:] - phi * residuals[:-1]
                    sigma_eps = np.std(eps)
                    var_res_hist = float(np.var(residuals))
                    print(f"   **AR(1)残差:** phi={phi:.3f}, sigma_eps={sigma_eps:.3f}, var_res_hist={var_res_hist:.3f}")
                    
                    # 模拟未来残差（可重复）并校准方差
                    rng = np.random.default_rng(self.random_seed)
                    linear_part = predicted_osf
                    res_sim = np.empty_like(linear_part, dtype=float)
                    res_sim[0] = residuals[-1] if len(residuals) > 0 else 0.0
                    for t in range(1, len(res_sim)):
                        res_sim[t] = phi * res_sim[t-1] + rng.normal(0.0, sigma_eps)
                    
                    # 方差校准到历史残差
                    var_res_sim = float(np.var(res_sim)) + 1e-12
                    k = np.sqrt(max(var_res_hist, 1e-12) / var_res_sim)
                    res_sim *= (k * self.osf_vol_scale)
                    print(f"   **残差方差校准:** scale={k:.3f}×vol_scale={self.osf_vol_scale:.2f}")
                    
                    predicted_osf = linear_part + res_sim

                    # —— 总方差校准（使预测OSF总方差接近历史OSF总方差） ——
                    try:
                        var_hist_total = float(np.var(clean_osf))
                        var_linear_pred = float(np.var(linear_part))
                        res_part = predicted_osf - linear_part
                        var_res_part = float(np.var(res_part)) + 1e-12
                        target_res_var = max(var_hist_total - var_linear_pred, 1e-12)
                        scale2 = np.sqrt(target_res_var / var_res_part)
                        res_part *= scale2
                        predicted_osf = linear_part + res_part
                        print(f"   **总方差校准:** scale2={scale2:.3f}")
                    except Exception as e:
                        print(f"⚠️ **总方差校准失败:** {e}")
            except Exception as e:
                print(f"⚠️ **AR(1)残差添加失败，原因:** {e}")
        else:
            # SSN无变化时，使用平均OSF
            predicted_osf = np.full_like(predicted_ssn, mean_osf, dtype=float)
            print(f"⚠️ **使用平均OSF值: {mean_osf:.6f}**")
        
        # 使用历史分位数范围裁剪，避免不合理饱和
        lo = float(np.quantile(clean_osf, 0.01))
        hi = float(np.quantile(clean_osf, 0.99))
        original_range = [predicted_osf.min(), predicted_osf.max()]
        predicted_osf = np.clip(predicted_osf, lo, hi)
        
        print(f"🎯 **OSF预测结果:**")
        print(f"   **约束前范围:** [{original_range[0]:.6f}, {original_range[1]:.6f}]")
        print(f"   **裁剪区间(1%-99%):** [{lo:.6f}, {hi:.6f}]")
        print(f"   **最终范围:** [{predicted_osf.min():.6f}, {predicted_osf.max():.6f}]")
        print(f"   **非零值数量:** {np.count_nonzero(predicted_osf)}/{len(predicted_osf)}")
        
        return predicted_osf

    def predict_solar_parameters_with_separate_polarity(self, solar_data, prediction_start_date, prediction_days=365):
        """
        **综合预测：其他参数基于11年周期，极性单独预测，OSF基于SSN预测**
        """
        print("🔮 **开始综合预测（极性单独处理，OSF基于SSN）...**")
        
        # 1) 极性
        polarity_predictions = self.predict_polarity_separately(solar_data, prediction_start_date, prediction_days)
        
        # 2) 构建模板
        historical_data = solar_data[solar_data['date'] < prediction_start_date]
        complete_cycles, smoothed_ssn = self.identify_complete_solar_cycles(historical_data)
        if len(complete_cycles) == 0:
            print("❌ **无法识别完整的太阳周期**")
            return None
        unified_phase, phase_templates = self.build_phase_templates(complete_cycles, [p for p in self.SOLAR_PARAMETERS if p not in ['polarity', 'daily_OSF']])
        
        # 3) 生成预测日期与相位
        prediction_dates = pd.date_range(start=prediction_start_date, periods=prediction_days, freq='D')
        prediction_results = {param: [] for param in self.SOLAR_PARAMETERS}
        days_since_last_cycle = (prediction_start_date - complete_cycles[-1]['end_date']).days
        avg_cycle_length = np.mean([c['length_days'] for c in complete_cycles])
        base_phase = days_since_last_cycle / avg_cycle_length
        
        # 4) 模板取值
        for day in range(prediction_days):
            cur_phase = (base_phase + day / avg_cycle_length) % 1.0
            idx = int(cur_phase * (len(unified_phase) - 1))
            for param in phase_templates:
                val = float(phase_templates[param][idx])
                # 约束
                if param == 'SSN':
                    val = max(0.0, val)
                elif param == 'wind_speed':
                    val = float(np.clip(val, 200, 800))
                elif param == 'HMF':
                    val = max(0.1, val)
                elif param == 'HCS_tilt':
                    val = float(np.clip(val, 0, 90))
                prediction_results[param].append(val)
        
        # 5) OSF 基于 SSN
        predicted_ssn = np.array(prediction_results['SSN'])
        predicted_osf = self.predict_osf_from_ssn(historical_data, predicted_ssn)
        prediction_results['daily_OSF'] = predicted_osf.tolist()
        
        # 6) 极性
        prediction_results['polarity'] = polarity_predictions
        
        # 7) 打包
        prediction_df = pd.DataFrame(prediction_results)
        prediction_df['date'] = prediction_dates
        
        print("\n📈 **预测完成统计:**")
        for p in self.SOLAR_PARAMETERS:
            if p == 'polarity':
                print(f"   **{p}:** 预测值 = {np.unique(prediction_df[p])}")
            else:
                print(f"   **{p}:** 范围=[{prediction_df[p].min():.2f}, {prediction_df[p].max():.2f}], 平均={prediction_df[p].mean():.2f}")
        
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
            'HCS_tilt': 'HCS Tilt Angle (degrees)',
            'daily_OSF': 'Daily Open Solar Flux'
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
                       color='red', linewidth=2, linestyle='--')
            else:
                ax.plot(prediction_results['date'], prediction_results[param], 
                       label='Prediction (11-year cycle)', 
                       color='red', linewidth=2, linestyle='--')

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
        plt.suptitle(f'Solar Parameter Prediction\nPolarity: 22-year cycle, OSF: based on SSN, Others: 11-year cycle\nPrediction: {prediction_start_date.strftime("%Y-%m-%d")} to 2033-01-01', 
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
    
    solar_data_file = '/home/phil/Files/lstmPaper/data/outputs/cycle_analysis/solar_physics_data_1985_2025_osf.csv'
    solar_data = predictor.load_solar_data(solar_data_file)
    
    prediction_start_date = datetime(2019, 10, 30)
    prediction_end_date = datetime(2033, 1, 1)
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
    output_file = output_dir + 'solar_physics_data_1985_2025_cycle_prediction_osf.csv'
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