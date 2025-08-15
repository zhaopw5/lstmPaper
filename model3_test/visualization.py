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



def plot_comprehensive_results(cosmic_data, train_losses, val_losses, test_predictions, test_actuals, test_dates, 
                              extended_pred_dates=None, extended_predictions=None):
    """Plot 4 key results in 2x2 layout"""
    print("Plotting comprehensive results...")
    
    # Create 2x2 layout for 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Subplot 1: Training loss
    axes[0, 0].plot(train_losses, label='Training Loss', color='blue', alpha=0.8)
    axes[0, 0].plot(val_losses, label='Validation Loss', color='red', alpha=0.8)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Process')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Subplot 2: Test prediction scatter
    axes[0, 1].scatter(test_actuals, test_predictions, alpha=0.6, color='green')
    axes[0, 1].plot([test_actuals.min(), test_actuals.max()], [test_actuals.min(), test_actuals.max()], 'r--', lw=2)
    r2 = r2_score(test_actuals, test_predictions)
    axes[0, 1].set_xlabel('Actual Values')
    axes[0, 1].set_ylabel('Predicted Values')
    axes[0, 1].set_title(f'Test Set Prediction (R²={r2:.4f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Subplot 3: Test time series
    axes[1, 0].plot(test_dates, test_actuals, label='Actual', alpha=0.8, linewidth=2, color='blue')
    axes[1, 0].plot(test_dates, test_predictions, label='Predicted', alpha=0.8, linewidth=2, color='red')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Helium Flux')
    axes[1, 0].set_title('Test Set Time Series')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Subplot 4: Complete time series (2011-2025)
    if extended_pred_dates is not None and extended_predictions is not None:
        obs_dates = cosmic_data['date YYYY-MM-DD']
        obs_flux = cosmic_data['helium_flux m^-2sr^-1s^-1GV^-1']
        
        axes[1, 1].plot(obs_dates, obs_flux, 'b-', label='Observed Data', linewidth=1.5, alpha=0.8)
        axes[1, 1].plot(extended_pred_dates, extended_predictions, 'r-', label='LSTM Predictions', linewidth=1.5, alpha=0.8)
        axes[1, 1].axvline(x=obs_dates.iloc[-1], color='green', linestyle='--', alpha=0.7, label='Observation End')
        axes[1, 1].set_title('Complete Time Series (2011-2025)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Helium Flux (m⁻²sr⁻¹s⁻¹GV⁻¹)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        axes[1, 1].xaxis.set_major_locator(mdates.YearLocator())
    else:
        # If no extended predictions, show model performance metrics
        residuals = test_actuals - test_predictions
        mse = np.mean(residuals**2)
        mae = np.mean(np.abs(residuals))
        r2 = r2_score(test_actuals, test_predictions)
        
        axes[1, 1].text(0.1, 0.8, f'MSE: {mse:.6f}', transform=axes[1, 1].transAxes, fontsize=14)
        axes[1, 1].text(0.1, 0.6, f'MAE: {mae:.6f}', transform=axes[1, 1].transAxes, fontsize=14)
        axes[1, 1].text(0.1, 0.4, f'R²: {r2:.6f}', transform=axes[1, 1].transAxes, fontsize=14)
        axes[1, 1].text(0.1, 0.2, f'Samples: {len(test_actuals)}', transform=axes[1, 1].transAxes, fontsize=14)
        axes[1, 1].set_title('Model Performance Metrics')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('LSTM_Prediction_Results.png', dpi=300, bbox_inches='tight')
