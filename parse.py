import json
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import glob
import json
import os
import matplotlib.dates as mdates
import seaborn as sns
from datetime import time



def parse_event_file(json_path: str) -> pd.DataFrame:
    """
    Lê o ficheiro JSON do evento e retorna um DataFrame com uma linha contendo:
    - triggerTs
    - Para cada estação (recorderName) e canal (T, R, V):
      - cf_peak
      - cf_rms
      - cf_value
      - fft_freq_peak (freq com maior amplitude)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    row = { 'triggerTs': data.get('triggerTs') }

    for ef in data.get('eventFiles', {}).values():
        rec = ef.get('recorderName')

        # Crest factor (cf)
        for cf in ef.get('df', {}).get('cf', []):
            ch = cf.get('chName')
            row[f'{rec}_{ch}_cf_peak'] = cf.get('peak')
            row[f'{rec}_{ch}_cf_rms'] = cf.get('rms')
            row[f'{rec}_{ch}_cf_value'] = cf.get('value')

        # FFT peaks: encontrar a frequência com maior amplitude
        for fft in ef.get('dfFft', {}).get('peak', []):
            ch = fft.get('chName')
            vals = fft.get('value', [])
            if vals:
                max_peak = max(vals, key=lambda x: x.get('ampl', 0))
                row[f'{rec}_{ch}_fft_freq_peak'] = max_peak.get('freq')
            else:
                row[f'{rec}_{ch}_fft_freq_peak'] = None

    # Construir DataFrame com uma única linha
    df = pd.DataFrame([row])
    return df

def parse_all_events(folder_path: str, pattern: str = "*.json") -> pd.DataFrame:
    """
    Percorre todos os ficheiros JSON na pasta especificada e retorna um DataFrame
    com uma linha por evento, concatenando os resultados de parse_event_file.

    Args:
        folder_path: Caminho para a pasta contendo os ficheiros JSON.
        pattern: Padrão de ficheiros a ler (por defeito, todos os .json).

    Retorna:
        pandas.DataFrame: DataFrame com todas as linhas de eventos.
    """
    # Encontrar todos os ficheiros que casam com o padrão
    search_path = os.path.join(folder_path, pattern)
    json_files = glob.glob(search_path)

    dfs = []
    for jf in json_files:
        try:
            df_event = parse_event_file(jf)
            dfs.append(df_event)
        except Exception as e:
            print(f"Aviso: não foi possível processar {jf}: {e}")

    # Concatenar todos os DataFrames (se houver algum)
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
    else:
        df_all = pd.DataFrame()

    return df_all



def convert_triggerTs_to_time(df: pd.DataFrame, column: str = 'triggerTs') -> pd.DataFrame:
    if column in df.columns:
        df = df.copy()
        df[column] = pd.to_datetime(df[column]).dt.time
    else:
        raise KeyError(f"Coluna '{column}' não encontrada no DataFrame")
    return df

def time_to_day_fraction(t: time) -> float:
    
    total_secs = t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
    return total_secs / 86400


def convert_triggerTs_to_float(df: pd.DataFrame, column: str = 'triggerTs') -> pd.DataFrame:
    if column in df.columns:
        df = df.copy()
        #df[column] = df[column].apply(time_to_day_fraction)
        
        df[column] = pd.to_datetime(df[column]).dt.time.apply(time_to_day_fraction)
        # # Tenta converter para float
        # df[column] = pd.to_numeric(df[column], errors='coerce').astype(float)
    else:
        raise KeyError(f"Coluna '{column}' não encontrada no DataFrame")
    return df

if __name__ == '__main__':
    # Exemplo de uso: ler todos os JSON de uma pasta chamada 'eventos'
    pasta = 'Data Json'
    df = parse_all_events(pasta)
    print(df)
    df_time = convert_triggerTs_to_time(df)
    print(df_time)


