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

# Picos máximo para comonentes T, R, V
max_T= df_consolidado_csv['T'].max()
max_R= df_consolidado_csv['R'].max()
max_V= df_consolidado_csv['V'].max()

# Encontrando tempos maximos nos picos
time_max_T =df_consolidado_csv['Time'][df_consolidado_csv['T'].idxmax()]
time_max_R =df_consolidado_csv['Time'][df_consolidado_csv['R'].idxmax()]
time_max_V =df_consolidado_csv['Time'][df_consolidado_csv['V'].idxmax()]

#Calculando a frequencia média para T
frq_media_T = 1/(time_max_T - df_consolidado_csv['Time'][0])
amplitute_max_T = max_T
#Calculando a frequencia média para R
freq_media_R = 1/(time_max_R - df_consolidado_csv['Time'][0])
amplitute_max_R = max_R
#Calculando a frequencia média para R
freq_media_V = 1/(time_max_V - df_consolidado_csv['Time'][0])
amplitute_max_V = max_V

#Plotagem dos gráficos
plt.figure(figsize=(10,6))
plt.scatter(frq_media_T, amplitute_max_T, color='black', label='Componente T')
plt.scatter(freq_media_R, amplitute_max_R, color='red', label='Componente R')
plt.scatter(freq_media_V, amplitute_max_V, color='blue', label='Componente V')
plt.xlabel('Frequência (Hz)')
plt.ylabel('Amplitude')
plt.title('Amplitude vs Frequência para as Componentes T, R e V')
plt.legend()
plt.grid(True)
plt.show()




