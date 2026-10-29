import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import VAR
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Paths are relative to the repository root, so the script runs from any folder
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def run_modeling_evaluation():
    print("Starting Phases 4 & 5: Modelling, Out-of-Sample Prediction and Forecasting...")
    
    # 1. Load data
    data_path = os.path.join(ROOT, 'data', 'norway_data.csv')
    df = pd.read_csv(data_path, parse_dates=['observation_date'], index_col='observation_date')
    
    n_obs = len(df)
    train_size = int(n_obs * 0.90)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # Log of the series
    train_log = np.log(train_df)
    test_log = np.log(test_df)
    full_log = np.log(df)
    
    # First difference of the logs (stationary series)
    train_log_diff = train_log.diff().dropna()
    
    # Results list for the final comparison table
    model_comparison = []
    predictions_dict = {'True': test_df['Share_Index']}
    
    # ==========================================
    # MODEL 1: SARIMAX (Univariate + Exogenous)
    # ==========================================
    print("\n--- Training Model 1: SARIMAX ---")
    
    # Use pmdarima to select the best SARIMAX model automatically.
    # Brent_Oil is an important exogenous variable, so it is included.
    # pmdarima selects the model based on AIC
    print("Selecting the optimal ARIMA order with pmdarima...")
    
    # Brent Oil as exogenous variable
    exog_train = train_log[['Brent_Oil']]
    exog_test = test_log[['Brent_Oil']]
    
    # Auto-ARIMA on log(Share_Index) with d=1 and exogenous regressor
    sarimax_auto = pm.auto_arima(
        train_log['Share_Index'],
        X=exog_train,
        d=1,
        seasonal=True,
        m=12,  # monthly seasonality
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False
    )
    
    order = sarimax_auto.order
    seasonal_order = sarimax_auto.seasonal_order
    print(f"Best SARIMAX order selected: ARIMA{order}x{seasonal_order}")
    
    # Fit the final model with statsmodels to get full diagnostics
    sarimax_model = SARIMAX(
        train_log['Share_Index'],
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    sarimax_results = sarimax_model.fit(disp=False)
    
    # Save the SARIMAX summary for the appendix
    with open(os.path.join(ROOT, 'report', 'tables', 'sarimax_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(str(sarimax_results.summary()))
        
    # Out-of-sample prediction (on the test set)
    # SARIMAX predicts in logs, then exp() is applied
    sarimax_pred_log = sarimax_results.predict(
        start=test_df.index[0],
        end=test_df.index[-1],
        exog=exog_test
    )
    sarimax_pred = np.exp(sarimax_pred_log)
    predictions_dict['SARIMAX'] = sarimax_pred
    
    # Error metrics
    sarimax_rmse = np.sqrt(mean_squared_error(test_df['Share_Index'], sarimax_pred))
    sarimax_mae = mean_absolute_error(test_df['Share_Index'], sarimax_pred)
    sarimax_mape = calculate_mape(test_df['Share_Index'], sarimax_pred)
    
    model_comparison.append({
        'Model': 'SARIMAX',
        'RMSE': sarimax_rmse,
        'MAE': sarimax_mae,
        'MAPE': sarimax_mape
    })
    
    # Future forecast (10 steps ahead)
    # Assume the exogenous variable (Brent Oil) stays at its last value in the future (or grows at its historical average rate)
    last_oil = test_log['Brent_Oil'].iloc[-1]
    exog_forecast = pd.DataFrame({'Brent_Oil': [last_oil] * 10}, 
                                 index=pd.date_range(start=test_df.index[-1] + pd.DateOffset(months=1), periods=10, freq='MS'))
    
    sarimax_forecast_log = sarimax_results.forecast(steps=10, exog=exog_forecast)
    sarimax_forecast = np.exp(sarimax_forecast_log)
    
    # ==========================================
    # MODEL 2: VAR (Vector Autoregression)
    # ==========================================
    print("\n--- Training Model 2: VAR ---")
    
    # VAR runs on the first differences of the logs (stationary)
    var_model = VAR(train_log_diff)
    
    # Optimal lag selection
    lag_selection = var_model.select_order(maxlags=12)
    best_lag = lag_selection.aic
    print(f"Best VAR lag (AIC criterion): {best_lag}")
    if best_lag == 0:
        best_lag = 1  # ensure at least 1 lag
        
    var_results = var_model.fit(best_lag)
    
    # Save the VAR summary
    with open(os.path.join(ROOT, 'report', 'tables', 'var_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(str(var_results.summary()))
        
    # Out-of-sample prediction for the VAR
    # Predict recursively over the test set (in log differences)
    # Then rebuild the levels from the last known training value
    test_inputs = train_log_diff.values[-best_lag:]
    var_preds_diff = var_results.forecast(test_inputs, steps=len(test_df))
    
    # Convert differenced predictions back to levels
    var_pred = []
    last_val_log = train_log['Share_Index'].iloc[-1]
    for d_pred in var_preds_diff[:, 0]:  # column 0 is Share_Index
        next_val_log = last_val_log + d_pred
        var_pred.append(np.exp(next_val_log))
        last_val_log = next_val_log
        
    var_pred = pd.Series(var_pred, index=test_df.index)
    predictions_dict['VAR'] = var_pred
    
    # Error metrics
    var_rmse = np.sqrt(mean_squared_error(test_df['Share_Index'], var_pred))
    var_mae = mean_absolute_error(test_df['Share_Index'], var_pred)
    var_mape = calculate_mape(test_df['Share_Index'], var_pred)
    
    model_comparison.append({
        'Model': 'VAR',
        'RMSE': var_rmse,
        'MAE': var_mae,
        'MAPE': var_mape
    })
    
    # Future forecast (10 steps ahead)
    test_all_diff = np.log(df).diff().dropna()
    var_inputs_forecast = test_all_diff.values[-best_lag:]
    var_forecast_diff = var_results.forecast(var_inputs_forecast, steps=10)
    
    var_forecast = []
    last_val_log = np.log(df['Share_Index']).iloc[-1]
    for d_pred in var_forecast_diff[:, 0]:
        next_val_log = last_val_log + d_pred
        var_forecast.append(np.exp(next_val_log))
        last_val_log = next_val_log
    
    var_forecast = pd.Series(var_forecast, index=exog_forecast.index)
    
    # ==========================================
    # MODEL 3: LSTM (Deep Learning)
    # ==========================================
    print("\n--- Training Model 3: LSTM ---")
    
    # Prepare the data for the recurrent neural network
    # Lags of the 3 log-differenced variables are used as inputs to predict the next log difference of Share_Index.
    full_diff = np.log(df).diff().dropna()
    train_diff_values = full_diff.iloc[:train_size-1].values  # -1 because of differencing
    
    lookback = 3 # use the previous 3 months
    
    def create_lstm_dataset(data, lookback):
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:i+lookback]) # lags of all 3 variables
            y.append(data[i+lookback, 0]) # predict the next change in Share_Index (column 0)
        return np.array(X), np.array(y)
        
    X_train, y_train = create_lstm_dataset(train_diff_values, lookback)
    
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    
    # LSTM model
    class LSTMForecaster(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers=1):
            super(LSTMForecaster, self).__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)
            
        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :]) # use the last hidden state
            return out
            
    lstm_model = LSTMForecaster(input_dim=3, hidden_dim=8)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.01)
    
    # Train the neural network
    epochs = 150
    lstm_model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = lstm_model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
            
    # Out-of-sample prediction
    # To predict dynamically and autonomously:
    # start from the training inputs and feed predictions back recursively
    lstm_model.eval()
    
    # Build the test dataset
    test_diff_values = full_diff.iloc[train_size-1-lookback:].values
    X_test, y_test = create_lstm_dataset(test_diff_values, lookback)
    X_test_t = torch.FloatTensor(X_test)
    
    with torch.no_grad():
        lstm_preds_diff = lstm_model(X_test_t).numpy().flatten()
        
    # Rebuild levels for the test set
    lstm_pred = []
    last_val_log = train_log['Share_Index'].iloc[-1]
    for d_pred in lstm_preds_diff[:len(test_df)]:
        next_val_log = last_val_log + d_pred
        lstm_pred.append(np.exp(next_val_log))
        last_val_log = next_val_log
        
    lstm_pred = pd.Series(lstm_pred, index=test_df.index)
    predictions_dict['LSTM'] = lstm_pred
    
    # Error metrics
    lstm_rmse = np.sqrt(mean_squared_error(test_df['Share_Index'], lstm_pred))
    lstm_mae = mean_absolute_error(test_df['Share_Index'], lstm_pred)
    lstm_mape = calculate_mape(test_df['Share_Index'], lstm_pred)
    
    model_comparison.append({
        'Model': 'LSTM',
        'RMSE': lstm_rmse,
        'MAE': lstm_mae,
        'MAPE': lstm_mape
    })
    
    # Future forecast (10 steps ahead)
    # Strictly recursive 10-step forecast:
    # The network needs all 3 variables as input but only column 0 is predicted,
    # so the other columns (exchange rate and oil) are assumed to stay at their last value.
    last_inputs = full_diff.iloc[-lookback:].values
    lstm_forecasts_diff = []
    current_input = last_inputs.copy()
    
    with torch.no_grad():
        for i in range(10):
            input_tensor = torch.FloatTensor(current_input).unsqueeze(0)
            pred_d = lstm_model(input_tensor).item()
            lstm_forecasts_diff.append(pred_d)
            # Update the inputs: drop the oldest row, add the new Share_Index prediction,
            # and assume zero change for USD_NOK and Brent_Oil
            new_row = np.array([pred_d, 0.0, 0.0])
            current_input = np.vstack([current_input[1:], new_row])
            
    lstm_forecast = []
    last_val_log = np.log(df['Share_Index']).iloc[-1]
    for d_pred in lstm_forecasts_diff:
        next_val_log = last_val_log + d_pred
        lstm_forecast.append(np.exp(next_val_log))
        last_val_log = next_val_log
        
    lstm_forecast = pd.Series(lstm_forecast, index=exog_forecast.index)
    
    # ==========================================
    # SAVE RESULTS AND COMPARISON
    # ==========================================
    comparison_df = pd.DataFrame(model_comparison)
    print("\n--- Model Performance Comparison ---")
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv(os.path.join(ROOT, 'report', 'tables', 'model_comparison.csv'), index=False)
    
    # Out-of-sample prediction plot (test period)
    plt.figure(figsize=(12, 6))
    plt.plot(test_df.index, test_df['Share_Index'], label='True Stock Index (Norway)', color='black', lw=2.2)
    plt.plot(test_df.index, sarimax_pred, label=f'SARIMAX (MAPE={sarimax_mape:.2f}%)', color='#d62728', lw=1.8, linestyle='-.')
    plt.plot(test_df.index, var_pred, label=f'VAR (MAPE={var_mape:.2f}%)', color='#1f77b4', lw=1.8, linestyle='--')
    plt.plot(test_df.index, lstm_pred, label=f'LSTM (MAPE={lstm_mape:.2f}%)', color='#2ca02c', lw=1.8, linestyle=':')
    plt.title('Out-of-Sample Predictions over the Test Period (2022-2026)', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Share Price Index (Norway)')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, 'report', 'plots', 'out_of_sample_comparison.png'), dpi=300)
    plt.close()
    
    # Future forecast plot (10 steps ahead)
    plt.figure(figsize=(12, 6))
    # Show the last 24 actual months for context
    history = df['Share_Index'].iloc[-24:]
    plt.plot(history.index, history, label='Actual History', color='black', lw=2.0)
    plt.plot(exog_forecast.index, sarimax_forecast, label='SARIMAX Forecast', color='#d62728', lw=2.0, marker='o', linestyle='-.')
    plt.plot(exog_forecast.index, var_forecast, label='VAR Forecast', color='#1f77b4', lw=2.0, marker='s', linestyle='--')
    plt.plot(exog_forecast.index, lstm_forecast, label='LSTM Forecast', color='#2ca02c', lw=2.0, marker='^', linestyle=':')
    plt.title('10-Month Future Forecast (April 2026 - January 2027)', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Share Price Index (Norway)')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, 'report', 'plots', 'future_forecast_comparison.png'), dpi=300)
    plt.close()
    
    # Save the forecasts to a table
    forecast_df = pd.DataFrame({
        'SARIMAX': sarimax_forecast,
        'VAR': var_forecast,
        'LSTM': lstm_forecast
    }, index=exog_forecast.index)
    forecast_df.to_csv(os.path.join(ROOT, 'report', 'tables', 'future_forecasts.csv'))
    
    print("\nModelling, evaluation and forecasting completed successfully!")
    print("High-resolution plots and tables were saved to the report/ folder.")

if __name__ == '__main__':
    run_modeling_evaluation()
