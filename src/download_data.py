import pandas as pd
import os

# Paths are relative to the repository root, so the script runs from any folder
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def download_data():
    print("Starting FRED data download...")
    
    # Direct CSV download URLs from FRED
    urls = {
        'Share_Index': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=SPASTT01NOM661N',
        'USD_NOK': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=EXNOUS',
        'Brent_Oil': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=POILBREUSDM'
    }
    
    dfs = {}
    for name, url in urls.items():
        print(f"Downloading {name} from FRED...")
        try:
            df = pd.read_csv(url, parse_dates=['observation_date'], index_col='observation_date')
            # FRED uses '.' for missing values, so convert to float and handle them
            col_name = df.columns[0]
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
            dfs[name] = df
            print(f"Success! {len(df)} observations retrieved from {df.index.min().date()} to {df.index.max().date()}.")
        except Exception as e:
            print(f"Error downloading {name}: {e}")
            return None
            
    # Merge the DataFrames on date (inner join to keep only complete observations)
    merged_df = dfs['Share_Index'].join(dfs['USD_NOK'], how='inner').join(dfs['Brent_Oil'], how='inner')
    
    # Handle missing values
    if merged_df.isnull().any().any():
        print("Missing values detected. Applying linear interpolation...")
        merged_df = merged_df.interpolate(method='linear')
        merged_df = merged_df.bfill().ffill() # extra safeguard for the endpoints
        
    merged_df.columns = ['Share_Index', 'USD_NOK', 'Brent_Oil']
    
    # Create the data folder if it does not exist
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(ROOT, 'data', 'norway_data.csv')
    merged_df.to_csv(output_path)
    print(f"\nData merged successfully and saved to: {os.path.abspath(output_path)}")
    print(f"Number of observations: {len(merged_df)}")
    print(f"Period: {merged_df.index.min().date()} to {merged_df.index.max().date()}")
    print("\nData Summary:")
    print(merged_df.describe())
    return merged_df

if __name__ == '__main__':
    download_data()
