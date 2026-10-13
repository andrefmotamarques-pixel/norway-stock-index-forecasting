import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron

# Paths are relative to the repository root, so the script runs from any folder
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_eda_stationarity():
    print("Iniciando Análise Exploratória e Testes de Estacionaridade...")
    
    # 1. Carregar dados
    data_path = os.path.join(ROOT, 'data', 'norway_data.csv')
    df = pd.read_csv(data_path, parse_dates=['observation_date'], index_col='observation_date')
    
    # Criar pasta para gráficos e tabelas se não existir
    os.makedirs(os.path.join(ROOT, 'report', 'plots'), exist_ok=True)
    os.makedirs(os.path.join(ROOT, 'report', 'tables'), exist_ok=True)
    
    # 2. Divisão de treino e teste (90% treino, 10% teste)
    n_obs = len(df)
    train_size = int(n_obs * 0.90)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    print(f"Total observações: {n_obs}")
    print(f"Treino (90%): {len(train_df)} (de {train_df.index.min().date()} a {train_df.index.max().date()})")
    print(f"Teste (10%): {len(test_df)} (de {test_df.index.min().date()} a {test_df.index.max().date()})")
    
    # 3. Estatísticas Descritivas (do conjunto de treino)
    desc_stats = train_df.describe().transpose()
    desc_stats['skewness'] = train_df.skew()
    desc_stats['kurtosis'] = train_df.kurt() # Curtose excessiva do pandas
    
    print("\nEstatísticas Descritivas (Treino):")
    print(desc_stats.to_string())
    desc_stats.to_csv(os.path.join(ROOT, 'report', 'tables', 'descriptive_stats.csv'))
    
    # 4. Gráficos das Séries Originais
    plt.figure(figsize=(12, 10))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, col in enumerate(df.columns):
        plt.subplot(3, 1, i+1)
        plt.plot(train_df.index, train_df[col], label=f'{col} (Treino)', color=colors[i], lw=1.8)
        plt.plot(test_df.index, test_df[col], label=f'{col} (Teste)', color=colors[i], linestyle='--', lw=1.5)
        plt.title(f'Série Temporal: {col}', fontsize=12, fontweight='bold')
        plt.ylabel('Valor')
        plt.legend(loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, 'report', 'plots', 'original_series.png'), dpi=300)
    plt.close()
    
    # 5. Análise de Estacionaridade (Nível vs Diferenças)
    def test_stationarity(series, name):
        # ADF Test
        adf_res = adfuller(series.dropna(), autolag='AIC', regression='ct') # Inclui constante e tendência linear
        
        # PP Test
        pp_res = PhillipsPerron(series.dropna(), trend='ct') # Constante e tendência
        
        # KPSS Test
        kpss_res = kpss(series.dropna(), regression='ct') # Constante e tendência
        
        return {
            'ADF_stat': adf_res[0], 'ADF_p': adf_res[1],
            'PP_stat': pp_res.stat, 'PP_p': pp_res.pvalue,
            'KPSS_stat': kpss_res[0], 'KPSS_p': kpss_res[1]
        }
        
    results_level = {}
    for col in train_df.columns:
        results_level[col] = test_stationarity(train_df[col], col)
        
    print("\n--- Resultados dos Testes de Estacionaridade (Em Nível) ---")
    for col, res in results_level.items():
        print(f"\nVariável: {col}")
        print(f"  ADF:  Stat={res['ADF_stat']:.4f}, p-value={res['ADF_p']:.4f} (H0: unit root, não-estacionária)")
        print(f"  PP:   Stat={res['PP_stat']:.4f}, p-value={res['PP_p']:.4f} (H0: unit root, não-estacionária)")
        print(f"  KPSS: Stat={res['KPSS_stat']:.4f}, p-value={res['KPSS_p']:.4f} (H0: estacionária, p < 0.05 rejeita H0)")
        
    # Salvar resultados em nível
    pd.DataFrame(results_level).transpose().to_csv(os.path.join(ROOT, 'report', 'tables', 'stationarity_level.csv'))
    
    # 6. Transformação (Log-Diferenças / Retornos Logarítmicos)
    # Para séries de preços (Share_Index, Brent_Oil) e cambial (USD_NOK), a log-diferença
    # é a melhor transformação para estabilizar a variância e representar taxas de crescimento/retornos.
    train_diff = np.log(train_df).diff().dropna()
    test_diff = np.log(df).diff().dropna().iloc[train_size-1:] # para manter o primeiro elemento do teste contínuo
    
    results_diff = {}
    for col in train_diff.columns:
        results_diff[col] = test_stationarity(train_diff[col], col)
        
    print("\n--- Resultados dos Testes de Estacionaridade (Em 1ª Diferença do Log) ---")
    for col, res in results_diff.items():
        print(f"\nVariável: {col} (log-diferenciada)")
        print(f"  ADF:  Stat={res['ADF_stat']:.4f}, p-value={res['ADF_p']:.4f}")
        print(f"  PP:   Stat={res['PP_stat']:.4f}, p-value={res['PP_p']:.4f}")
        print(f"  KPSS: Stat={res['KPSS_stat']:.4f}, p-value={res['KPSS_p']:.4f}")
        
    pd.DataFrame(results_diff).transpose().to_csv(os.path.join(ROOT, 'report', 'tables', 'stationarity_diff.csv'))
    
    # Gráficos das Séries Diferenciadas
    plt.figure(figsize=(12, 10))
    for i, col in enumerate(train_diff.columns):
        plt.subplot(3, 1, i+1)
        plt.plot(train_diff.index, train_diff[col], label=f'{col} (Treino)', color=colors[i], lw=1.2)
        plt.plot(test_diff.index[1:], test_diff[col].iloc[1:], label=f'{col} (Teste)', color=colors[i], linestyle='--', lw=1.0)
        plt.title(f'Série Temporal (1ª Diferença do Log): {col}', fontsize=12, fontweight='bold')
        plt.ylabel('Log-Retorno')
        plt.legend(loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, 'report', 'plots', 'differenced_series.png'), dpi=300)
    plt.close()
    
    print("\nAnálise terminada com sucesso! Gráficos e tabelas salvos na pasta report/")

if __name__ == '__main__':
    run_eda_stationarity()
