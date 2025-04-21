import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

caminho_pasta_csv = r'C:\Users\mathe\OneDrive\Área de Trabalho\Estágio\Projeto 2\Data csv'

lista_csv = [] # Cria uma lista vazia para armazenar os dados
arquivos_csv = glob.glob(os.path.join(caminho_pasta_csv, "*.csv")) # Pega todos os arquivos da minhas pasta com formato .csv

for arquivo in arquivos_csv: # Para cada arquivo na minha pasta faça:
    try: # Tente:
        df_csv = pd.read_csv(arquivo) # Tranforma cada arquivo lido em um dataframe 
        lista_csv.append(df_csv) # Adiciona ao final da minha lista vazia.
    except Exception as e:
        print(f'Erro ao tentar ler esse arquivo aqui: {e}') # Caso ele nao consiga pegar me de um erro e me diga em qual arquivo não conseguiu

#Consolidando os arquivos que leu em um só
if lista_csv:
    df_consolidado_csv = pd.concat(lista_csv, ignore_index=True) # Concat concatenar os arquivos na minha lista
    print('Consolidação feita com sucesso! DATA: ')
else:
    print('Aqui deu ruim meu chapa  :( ')


print(df_consolidado_csv.head(3))

df= df_consolidado_csv
componentes = ['T', 'R', 'V']

tempo = df['Time']
dt = tempo[1] - tempo[0]
fs = 1 / dt  # Frequência de amostragem
print(f'Frequência de amostragem: {fs:.2f} Hz')

# Layout dos gráficos
plt.figure(figsize=(12, 9))

for i, comp in enumerate(componentes):
    sinal = df[comp]
    N = len(sinal)

    # FFT
    fft_result = np.fft.fft(sinal)
    freqs = np.fft.fftfreq(N, d=1/fs)
    mask = freqs >= 0
    freqs_pos = freqs[mask]
    fft_magnitude = np.abs(fft_result[mask]) * 2 / N

    # Ponto de pico
    max_magnitude = np.max(fft_magnitude)
    freq_max = freqs_pos[np.argmax(fft_magnitude)]
    print(f'Componente {comp} -> Máxima magnitude: {max_magnitude:.5f} em {freq_max:.2f} Hz')

    # Gráfico do sinal no tempo (onda suave)
    plt.subplot(3, 2, 2*i + 1)
    plt.plot(tempo, sinal, color='black', linewidth=1)
    plt.title(f'Sinal no Tempo - Componente {comp}')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Gráfico da FFT (linha contínua e suave)
    plt.subplot(3, 2, 2*i + 2)
    plt.plot(freqs_pos, fft_magnitude, color='black', linewidth=1)
    plt.title(f'Espectro de Frequência - Componente {comp}')
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)

plt.tight_layout()
plt.show()