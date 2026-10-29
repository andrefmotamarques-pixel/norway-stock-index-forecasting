import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron

# Paths are relative to the repository root, so the script runs from any folder
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_eda_stationarity():
    print("Starting Exploratory Analysis and Stationarity Tests...")
    
    # 1. Load data
    data_path = os.path.join(ROOT, 'data', 'norway_data.csv')
    df = pd.read_csv(data_path, parse_dates=['observation_date'], index_col='observation_date')
    
    # Create the plots and tables folders if they do not exist
    os.makedirs(os.path.join(ROOT, 'report', 'plots'), exist_ok=True)
    os.makedirs(os.path.join(ROOT, 'report', 'tables'), exist_ok=True)
    
    # 2. Train/test split (90% train, 10% test)
    n_obs = len(df)
    train_size = int(n_obs * 0.90)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    print(f"Total observations: {n_obs}")
    print(f"Train (90%): {len(train_df)} ({train_df.index.min().date()} to {train_df.index.max().date()})")
    print(f"Test (10%): {len(test_df)} ({test_df.index.min().date()} to {test_df.index.max().date()})")
    
    # 3. Descriptive Statistics (training set)
    desc_stats = train_df.describe().transpose()
    desc_stats['skewness'] = train_df.skew()
    desc_stats['kurtosis'] = train_df.kurt() # pandas returns excess kurtosis
    
    print("\nDescriptive Statistics (Train):")
    print(desc_stats.to_string())
    desc_stats.to_csv(os.path.join(ROOT, 'report', 'tables', 'descriptive_stats.csv'))
    
    # 4. Plots of the Original Series
    plt.figure(figsize=(12, 10))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, col in enumerate(df.columns):
        plt.subplot(3, 1, i+1)
        plt.plot(train_df.index, train_df[col], label=f'{col} (Train)', color=colors[i], lw=1.8)
        plt.plot(test_df.index, test_df[col], label=f'{col} (Test)', color=colors[i], linestyle='--', lw=1.5)
        plt.title(f'Time Series: {col}', fontsize=12, fontweight='bold')
        plt.ylabel('Value')
        plt.legend(loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, 'report', 'plots', 'original_series.png'), dpi=300)
    plt.close()
    
    # 5. Stationarity Analysis (Levels vs Differences)
    def test_stationarity(series, name):
        # ADF Test
        adf_res = adfuller(series.dropna(), autolag='AIC', regression='ct') # Includes constant and linear trend
        
        # PP Test
        pp_res = PhillipsPerron(series.dropna(), trend='ct') # Constant and trend
        
        # KPSS Test
        kpss_res = kpss(series.dropna(), regression='ct') # Constant and trend
        
        return {
            'ADF_stat': adf_res[0], 'ADF_p': adf_res[1],
            'PP_stat': pp_res.stat, 'PP_p': pp_res.pvalue,
            'KPSS_stat': kpss_res[0], 'KPSS_p': kpss_res[1]
        }
        
    results_level = {}
    for col in train_df.columns:
        results_level[col] = test_stationarity(train_df[col], col)
        
    print("\n--- Stationarity Test Results (Levels) ---")
    for col, res in results_level.items():
        print(f"\nVariable: {col}")
        print(f"  ADF:  Stat={res['ADF_stat']:.4f}, p-value={res['ADF_p']:.4f} (H0: unit root, non-stationary)")
        print(f"  PP:   Stat={res['PP_stat']:.4f}, p-value={res['PP_p']:.4f} (H0: unit root, non-stationary)")
        print(f"  KPSS: Stat={res['KPSS_stat']:.4f}, p-value={res['KPSS_p']:.4f} (H0: stationary, p < 0.05 rejects H0)")
        
    # Save level results
    pd.DataFrame(results_level).transpose().to_csv(os.path.join(ROOT, 'report', 'tables', 'stationarity_level.csv'))
    
    # 6. Transformation (Log Differences / Log Returns)
    # For price series (Share_Index, Brent_Oil) and the exchange rate (USD_NOK), the log difference
    # is the best transformation to stabilise the variance and represent growth rates/returns.
    train_diff = np.log(train_df).diff().dropna()
    test_diff = np.log(df).diff().dropna().iloc[train_size-1:] # keeps the first test element continuous with the training set
    
    results_diff = {}
    for col in train_diff.columns:
        results_diff[col] = test_stationarity(train_diff[col], col)
        
    print("\n--- Stationarity Test Results (First Difference of Log) ---")
    for col, res in results_diff.items():
        print(f"\nVariable: {col} (log-differenced)")
        print(f"  ADF:  Stat={res['ADF_stat']:.4f}, p-value={res['ADF_p']:.4f}")
        print(f"  PP:   Stat={res['PP_stat']:.4f}, p-value={res['PP_p']:.4f}")
        print(f"  KPSS: Stat={res['KPSS_stat']:.4f}, p-value={res['KPSS_p']:.4f}")
        
    pd.DataFrame(results_diff).transpose().to_csv(os.path.join(ROOT, 'report', 'tables', 'stationarity_diff.csv'))
    
    # Plots of the Differenced Series
    plt.figure(figsize=(12, 10))
    for i, col in enumerate(train_diff.columns):
        plt.subplot(3, 1, i+1)
        plt.plot(train_diff.index, train_diff[col], label=f'{col} (Train)', color=colors[i], lw=1.2)
        plt.plot(test_diff.index[1:], test_diff[col].iloc[1:], label=f'{col} (Test)', color=colors[i], linestyle='--', lw=1.0)
        plt.title(f'Time Series (First Difference of Log): {col}', fontsize=12, fontweight='bold')
        plt.ylabel('Log Return')
        plt.legend(loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, 'report', 'plots', 'differenced_series.png'), dpi=300)
    plt.close()
    
    print("\nAnalysis completed successfully! Plots and tables saved to the report/ folder")

if __name__ == '__main__':
    run_eda_stationarity()
