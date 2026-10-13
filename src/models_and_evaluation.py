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

# Configuração de sementes para reprodutibilidade
np.random.seed(42)
torch.manual_seed(42)

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def run_modeling_evaluation():
    print("Iniciando Fase 4 & 5: Modelação, Previsão Out-of-Sample e Forecast...")
    
    # 1. Carregar dados
    data_path = os.path.join(ROOT, 'data', 'norway_data.csv')
    df = pd.read_csv(data_path, parse_dates=['observation_date'], index_col='observation_date')
    
    n_obs = len(df)
    train_size = int(n_obs * 0.90)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # Logs das séries
    train_log = np.log(train_df)
    test_log = np.log(test_df)
    full_log = np.log(df)
    
    # 1ª Diferença dos Logs (Séries Estacionárias)
    train_log_diff = train_log.diff().dropna()
    
    # Lista de resultados para a tabela final de comparação
    model_comparison = []
    predictions_dict = {'True': test_df['Share_Index']}
    
    # ==========================================
    # MODELO 1: SARIMAX (Univariado + Exógenas)
    # ==========================================
    print("\n--- A treinar Modelo 1: SARIMAX ---")
    
    # Vamos usar pmdarima para selecionar o melhor modelo SARIMAX automaticamente.
    # Como Brent_Oil é uma variável exógena importante, vamos usá-la.
    # pmdarima seleciona o modelo baseado no AIC
    print("A selecionar ordem ARIMA ótima com pmdarima...")
    
    # Brent Oil exógena
    exog_train = train_log[['Brent_Oil']]
    exog_test = test_log[['Brent_Oil']]
    
    # Auto-ARIMA no log(Share_Index) com d=1 e exog
    sarimax_auto = pm.auto_arima(
        train_log['Share_Index'],
        X=exog_train,
        d=1,
        seasonal=True,
        m=12,  # sazonalidade mensal
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False
    )
    
    order = sarimax_auto.order
    seasonal_order = sarimax_auto.seasonal_order
    print(f"Melhor ordem SARIMAX selecionada: ARIMA{order}x{seasonal_order}")
    
    # Ajustar o modelo final com statsmodels para ter diagnósticos completos
    sarimax_model = SARIMAX(
        train_log['Share_Index'],
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    sarimax_results = sarimax_model.fit(disp=False)
    
    # Salvar resumo do SARIMAX para anexo
    with open(os.path.join(ROOT, 'report', 'tables', 'sarimax_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(str(sarimax_results.summary()))
        
    # Previsão out-of-sample (no conjunto de teste)
    # SARIMAX prevê em log, depois fazemos exp()
    sarimax_pred_log = sarimax_results.predict(
        start=test_df.index[0],
        end=test_df.index[-1],
        exog=exog_test
    )
    sarimax_pred = np.exp(sarimax_pred_log)
    predictions_dict['SARIMAX'] = sarimax_pred
    
    # Métricas de erro
    sarimax_rmse = np.sqrt(mean_squared_error(test_df['Share_Index'], sarimax_pred))
    sarimax_mae = mean_absolute_error(test_df['Share_Index'], sarimax_pred)
    sarimax_mape = calculate_mape(test_df['Share_Index'], sarimax_pred)
    
    model_comparison.append({
        'Model': 'SARIMAX',
        'RMSE': sarimax_rmse,
        'MAE': sarimax_mae,
        'MAPE': sarimax_mape
    })
    
    # Forecast futuro (10 passos à frente)
    # Assumimos que a exógena (Brent Oil) mantém o seu último valor no futuro (ou cresce ao valor médio histórico)
    last_oil = test_log['Brent_Oil'].iloc[-1]
    exog_forecast = pd.DataFrame({'Brent_Oil': [last_oil] * 10}, 
                                 index=pd.date_range(start=test_df.index[-1] + pd.DateOffset(months=1), periods=10, freq='MS'))
    
    sarimax_forecast_log = sarimax_results.forecast(steps=10, exog=exog_forecast)
    sarimax_forecast = np.exp(sarimax_forecast_log)
    
    # ==========================================
    # MODELO 2: VAR (Vetor Autorregressivo)
    # ==========================================
    print("\n--- A treinar Modelo 2: VAR ---")
    
    # VAR corre sobre as primeiras diferenças dos logs (estacionárias)
    var_model = VAR(train_log_diff)
    
    # Seleção de lag ótima
    lag_selection = var_model.select_order(maxlags=12)
    best_lag = lag_selection.aic
    print(f"Melhor lag para VAR (critério AIC): {best_lag}")
    if best_lag == 0:
        best_lag = 1  # garantir pelo menos 1 lag
        
    var_results = var_model.fit(best_lag)
    
    # Salvar resumo do VAR
    with open(os.path.join(ROOT, 'report', 'tables', 'var_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(str(var_results.summary()))
        
    # Previsão out-of-sample para o VAR
    # Precisamos de prever recursivamente no conjunto de teste (em log-diferenças)
    # Depois, reconstruímos os níveis usando o último valor conhecido do treino
    test_inputs = train_log_diff.values[-best_lag:]
    var_preds_diff = var_results.forecast(test_inputs, steps=len(test_df))
    
    # Converter previsões diff de volta para nível
    var_pred = []
    last_val_log = train_log['Share_Index'].iloc[-1]
    for d_pred in var_preds_diff[:, 0]:  # a coluna 0 é Share_Index
        next_val_log = last_val_log + d_pred
        var_pred.append(np.exp(next_val_log))
        last_val_log = next_val_log
        
    var_pred = pd.Series(var_pred, index=test_df.index)
    predictions_dict['VAR'] = var_pred
    
    # Métricas de erro
    var_rmse = np.sqrt(mean_squared_error(test_df['Share_Index'], var_pred))
    var_mae = mean_absolute_error(test_df['Share_Index'], var_pred)
    var_mape = calculate_mape(test_df['Share_Index'], var_pred)
    
    model_comparison.append({
        'Model': 'VAR',
        'RMSE': var_rmse,
        'MAE': var_mae,
        'MAPE': var_mape
    })
    
    # Forecast futuro (10 passos à frente)
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
    # MODELO 3: LSTM (Deep Learning)
    # ==========================================
    print("\n--- A treinar Modelo 3: LSTM ---")
    
    # Preparar os dados para a rede neuronal recorrente
    # Usaremos lags das 3 variáveis log-diferenciadas como inputs e preveremos o próximo log-diferença de Share_Index.
    full_diff = np.log(df).diff().dropna()
    train_diff_values = full_diff.iloc[:train_size-1].values  # -1 por causa da diferença
    
    lookback = 3 # usar 3 meses anteriores
    
    def create_lstm_dataset(data, lookback):
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:i+lookback]) # lags de todas as 3 variáveis
            y.append(data[i+lookback, 0]) # prever a próxima variação do Share_Index (coluna 0)
        return np.array(X), np.array(y)
        
    X_train, y_train = create_lstm_dataset(train_diff_values, lookback)
    
    # Converter para tensores do PyTorch
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    
    # Modelo LSTM
    class LSTMForecaster(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers=1):
            super(LSTMForecaster, self).__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)
            
        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :]) # usar o último estado oculto
            return out
            
    lstm_model = LSTMForecaster(input_dim=3, hidden_dim=8)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.01)
    
    # Treino da Rede Neuronal
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
            
    # Previsão out-of-sample
    # Para prever dinamicamente e de forma autónoma:
    # Usaremos os inputs de treino e alimentamos a previsão recursivamente
    lstm_model.eval()
    
    # Vamos criar o dataset de teste
    test_diff_values = full_diff.iloc[train_size-1-lookback:].values
    X_test, y_test = create_lstm_dataset(test_diff_values, lookback)
    X_test_t = torch.FloatTensor(X_test)
    
    with torch.no_grad():
        lstm_preds_diff = lstm_model(X_test_t).numpy().flatten()
        
    # Reconstruir em níveis para o conjunto de teste
    lstm_pred = []
    last_val_log = train_log['Share_Index'].iloc[-1]
    for d_pred in lstm_preds_diff[:len(test_df)]:
        next_val_log = last_val_log + d_pred
        lstm_pred.append(np.exp(next_val_log))
        last_val_log = next_val_log
        
    lstm_pred = pd.Series(lstm_pred, index=test_df.index)
    predictions_dict['LSTM'] = lstm_pred
    
    # Métricas de erro
    lstm_rmse = np.sqrt(mean_squared_error(test_df['Share_Index'], lstm_pred))
    lstm_mae = mean_absolute_error(test_df['Share_Index'], lstm_pred)
    lstm_mape = calculate_mape(test_df['Share_Index'], lstm_pred)
    
    model_comparison.append({
        'Model': 'LSTM',
        'RMSE': lstm_rmse,
        'MAE': lstm_mae,
        'MAPE': lstm_mape
    })
    
    # Forecast futuro (10 passos à frente)
    # Previsão recursiva estrita a 10 passos:
    # Como a rede precisa das 3 variáveis para o input, e só prevemos a coluna 0,
    # vamos assumir que as outras colunas (taxa de câmbio e petróleo) se mantêm estáveis no último valor.
    last_inputs = full_diff.iloc[-lookback:].values
    lstm_forecasts_diff = []
    current_input = last_inputs.copy()
    
    with torch.no_grad():
        for i in range(10):
            input_tensor = torch.FloatTensor(current_input).unsqueeze(0)
            pred_d = lstm_model(input_tensor).item()
            lstm_forecasts_diff.append(pred_d)
            # Atualizar os inputs: remove o mais antigo, adiciona a nova previsão de Share_Index,
            # e assume variação zero (0) para USD_NOK e Brent_Oil
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
    # SALVAR RESULTADOS E COMPARAÇÃO
    # ==========================================
    comparison_df = pd.DataFrame(model_comparison)
    print("\n--- Comparação de Desempenho dos Modelos ---")
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv(os.path.join(ROOT, 'report', 'tables', 'model_comparison.csv'), index=False)
    
    # Gráfico de Previsão Out-of-Sample (Teste)
    plt.figure(figsize=(12, 6))
    plt.plot(test_df.index, test_df['Share_Index'], label='True Stock Index (Noruega)', color='black', lw=2.2)
    plt.plot(test_df.index, sarimax_pred, label=f'SARIMAX (MAPE={sarimax_mape:.2f}%)', color='#d62728', lw=1.8, linestyle='-.')
    plt.plot(test_df.index, var_pred, label=f'VAR (MAPE={var_mape:.2f}%)', color='#1f77b4', lw=1.8, linestyle='--')
    plt.plot(test_df.index, lstm_pred, label=f'LSTM (MAPE={lstm_mape:.2f}%)', color='#2ca02c', lw=1.8, linestyle=':')
    plt.title('Previsão Out-of-Sample no Período de Teste (2022-2026)', fontsize=14, fontweight='bold')
    plt.xlabel('Data')
    plt.ylabel('Índice de Ações (Noruega)')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, 'report', 'plots', 'out_of_sample_comparison.png'), dpi=300)
    plt.close()
    
    # Gráfico de Forecast Futuro (10 passos à frente)
    plt.figure(figsize=(12, 6))
    # Mostrar últimos 24 meses reais para contexto
    history = df['Share_Index'].iloc[-24:]
    plt.plot(history.index, history, label='Histórico Real', color='black', lw=2.0)
    plt.plot(exog_forecast.index, sarimax_forecast, label='SARIMAX Forecast', color='#d62728', lw=2.0, marker='o', linestyle='-.')
    plt.plot(exog_forecast.index, var_forecast, label='VAR Forecast', color='#1f77b4', lw=2.0, marker='s', linestyle='--')
    plt.plot(exog_forecast.index, lstm_forecast, label='LSTM Forecast', color='#2ca02c', lw=2.0, marker='^', linestyle=':')
    plt.title('Forecast Futuro a 10 Meses (Abril 2026 - Janeiro 2027)', fontsize=14, fontweight='bold')
    plt.xlabel('Data')
    plt.ylabel('Índice de Ações (Noruega)')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, 'report', 'plots', 'future_forecast_comparison.png'), dpi=300)
    plt.close()
    
    # Salvar forecasts numa tabela
    forecast_df = pd.DataFrame({
        'SARIMAX': sarimax_forecast,
        'VAR': var_forecast,
        'LSTM': lstm_forecast
    }, index=exog_forecast.index)
    forecast_df.to_csv(os.path.join(ROOT, 'report', 'tables', 'future_forecasts.csv'))
    
    print("\nModelação, avaliação e forecast executados com total sucesso!")
    print("Gráficos e tabelas foram guardados na pasta report/ com alta qualidade.")

if __name__ == '__main__':
    run_modeling_evaluation()
