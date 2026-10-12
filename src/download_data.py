import pandas as pd
import os

def download_data():
    print("Iniciando download dos dados da FRED...")
    
    # URLs diretas para download de CSV da FRED
    urls = {
        'Share_Index': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=SPASTT01NOM661N',
        'USD_NOK': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=EXNOUS',
        'Brent_Oil': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=POILBREUSDM'
    }
    
    dfs = {}
    for name, url in urls.items():
        print(f"A descarregar {name} da FRED...")
        try:
            df = pd.read_csv(url, parse_dates=['observation_date'], index_col='observation_date')
            # FRED usa '.' para valores em falta. Vamos converter para float e tratar
            col_name = df.columns[0]
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
            dfs[name] = df
            print(f"Sucesso! {len(df)} observações obtidas de {df.index.min().date()} a {df.index.max().date()}.")
        except Exception as e:
            print(f"Erro ao descarregar {name}: {e}")
            return None
            
    # Fundir os DataFrames com base na data (inner join para garantir observações completas)
    merged_df = dfs['Share_Index'].join(dfs['USD_NOK'], how='inner').join(dfs['Brent_Oil'], how='inner')
    
    # Tratar valores em falta
    if merged_df.isnull().any().any():
        print("Valores em falta detetados. A aplicar interpolação linear...")
        merged_df = merged_df.interpolate(method='linear')
        merged_df = merged_df.bfill().ffill() # garantias adicionais para os extremos
        
    merged_df.columns = ['Share_Index', 'USD_NOK', 'Brent_Oil']
    
    # Criar pasta data se não existir
    os.makedirs('data', exist_ok=True)
    
    # Salvar em CSV
    output_path = os.path.join('data', 'norway_data.csv')
    merged_df.to_csv(output_path)
    print(f"\nDados combinados com sucesso e salvos em: {os.path.abspath(output_path)}")
    print(f"Número de observações: {len(merged_df)}")
    print(f"Período: de {merged_df.index.min().date()} a {merged_df.index.max().date()}")
    print("\nResumo dos Dados:")
    print(merged_df.describe())
    return merged_df

if __name__ == '__main__':
    download_data()
