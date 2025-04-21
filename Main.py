import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import glob
import json
import os
import matplotlib.dates as mdates


# Leitura dos dados csv's consolidação de todos.
caminho_pasta_csv = r'C:\Users\mathe\OneDrive\Área de Trabalho\Estágio\Projeto 2\Data csv'
caminho_pasta_freq = r'C:\Users\mathe\OneDrive\Área de Trabalho\Estágio\Projeto 2\Data freq'

lista_csv = [] # Cria uma lista vazia para armazenar os dados
arquivos_csv = glob.glob(os.path.join(caminho_pasta_csv, "*.csv")) # Pega todos os arquivos da minhas pasta com formato .csv

for arquivo in arquivos_csv: # Para cada arquivo na minha pasta faça:
    try: # Tente:
        df_csv = pd.read_csv(arquivo) # Tranforma cada arquivo lido em um dataframe 
        lista_csv.append(df_csv) # Adiciona ao final da minha lista vazia.
    except Exception as e:
        print(f'Erro ao tentar ler esse arquivo aqui: {e}') # Caso ele nao consiga pegar me de um erro e me diga em qual arquivo não conseguiu

if lista_csv:
    df_consolidado_csv = pd.concat(lista_csv, ignore_index=True)
    print('Consolidação feita com sucesso! DATA: ')
else:
    print('Aqui deu ruim meu chapa  :( ')


# Leitura dos dados e consolidação dos dados freq.
list_freq = [] # Cria uma lista vazia para armazenar os dados.
arquivos_freq = glob.glob(os.path.join(caminho_pasta_freq, "*.csv")) 

for arquivo in arquivos_freq: # Para cada Arquivo em 
    try:
        df_freq = pd.read_csv(arquivo)
        list_freq.append(df_freq)
    except Exception as e:
        print(f'Erro ao tentar ler isso aqui meu chapa')


if lista_csv:
    df_consolidado_freq = pd.concat(list_freq, ignore_index=True)
    print('Consolidação feita com sucesso! FREQ: ')
else:
    print('Deu ruim meu chapa! : (')


# Lista que vai armazenar os dados consolidados
lista_dfs = []

# Caminho da pasta com os JSONs
caminho_pasta = "Data Json"

arquivos_json = glob.glob(os.path.join(caminho_pasta, "*.json"))

# Loop por todos os arquivos .json da pasta
for file in glob.glob(os.path.join(caminho_pasta, "*.json")):
    with open(file, "r", encoding="utf-8") as f:
        dados = json.load(f)

        for recorder_id, event_data in dados['eventFiles'].items():
            # Informações principais do evento e da estação.
            base_info = {
                'id_global': dados.get('id'), # Pega a coluna de ID.
                'triggerTs_global': dados.get('triggerTs'), # Pega a coluna de Triggers
                'data_arquivo': os.path.basename(file)  # nome do arquivo json/ apenas para rastreio
            }

            # Acessa os dados por canal (ex: canais T, R, V) dentro da chave 'cf'
            if 'df' in event_data and 'cf' in event_data['df']:
                # Para cada canal (T, R, V) nos dados temporais
                for canal in event_data['df']['cf']:
                    # Cria uma nova linha com as informações básicas
                    linha = base_info.copy()
                    
                    # Adiciona os dados do domínio do tempo
                    linha['canal'] = canal['chName']
                    linha['peak_time'] = canal['peak']
                    linha['rms_time'] = canal['rms']
                    linha['value_time'] = canal['value']
                    
                    # Inicializa as colunas de frequência como NaN
                    for i in range(5):
                        linha[f'freq_{i}'] = np.nan
                        linha[f'ampl_{i}'] = np.nan
                    
                    # Adiciona os dados de frequência se existirem
                    if 'dfFft' in event_data and 'peak' in event_data['dfFft']:
                        # Encontra os dados de frequência para este canal específico
                        freq_data = next((item for item in event_data['dfFft']['peak'] 
                                        if item['chName'] == canal['chName']), None)
                        
                        if freq_data:
                            # Pega os 5 principais picos de frequência
                            for i, pico in enumerate(freq_data['value'][:5]):
                                linha[f'freq_{i}'] = pico['freq']
                                linha[f'ampl_{i}'] = pico['ampl']
                    
                    # Adiciona a linha completa à lista
                    lista_dfs.append(linha)

# Cria o DataFrame consolidado
df_consolidado_json = pd.DataFrame(lista_dfs)

print(df_consolidado_json.shape)
print(df_consolidado_csv.shape)
print(df_consolidado_freq.shape)
colunas_analise = df_consolidado_json.loc[:, ['canal'] + list(df_consolidado_json.loc[:, 'peak_time':].columns)]


# Mostra as primeiras linhas
print(colunas_analise.head())

#Limpeza Pré processamento de dados para cluster

colunas_para_cluster = ['peak_time', 'rms_time'] + [f'freq_{i}' for i in range(5)] + [f'ampl_{i}' for i in range(5)]
df_cluster = df_consolidado_json[colunas_para_cluster].copy()
df_cluster.fillna(df_cluster.median(numeric_only=True), inplace=True) # Removedno valores nulose  preenchendo com a mediana

# Normalizar dados para o k-Means.
scaler = StandardScaler()
X = scaler.fit_transform(df_cluster)
#Testando primeiro com 3 clusters

kmeans = KMeans(n_clusters=3, random_state=42)
df_consolidado_json['cluster'] = kmeans.fit_predict(X)

# Fazendo PCA e Plotando.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df_consolidado_json['cluster'], cmap='viridis')
plt.title('Distribuição de Eventos Sísmicos com Base em Padrões de Frequência e Intensidade')
plt.xlabel('Padrão de Intensidade Sísmica')
plt.ylabel('Variação de Frequência / Amplitude')
plt.colorbar(label='Cluster')
plt.grid(True)
#plt.show()




# Conversão de data e hora dde colunas triggerTs_global

df_consolidado_json['triggerTs_global'] = pd.to_datetime(df_consolidado_json['triggerTs_global'])

# Adiciona colunas auxiliares para facilitar
# Usa o mesmo colormap que o scatter de KMeans
cmap = plt.get_cmap('viridis')
cores = df_consolidado_json['cluster'].map(lambda x: cmap(x / df_consolidado_json['cluster'].nunique()))
df_consolidado_json['hora'] = df_consolidado_json['triggerTs_global'].dt.hour

plt.figure(figsize=(10, 5))
for cluster in sorted(df_consolidado_json['cluster'].unique()):
    dados_cluster = df_consolidado_json[df_consolidado_json['cluster'] == cluster]
    plt.hist(dados_cluster['hora'], bins=24, alpha=0.6, label=f'Cluster {cluster}', color=cmap(cluster / df_consolidado_json['cluster'].nunique()))

plt.title('Distribuição de Eventos por Hora do Dia (por Cluster)')
plt.xlabel('Hora do Dia')
plt.ylabel('Número de Eventos')
plt.legend()
plt.grid(True)
plt.show()
 
# Acima, testando meu codigo.

print(f'Dados para visão de variável colunas de cluster: \n {colunas_para_cluster}')

# Identificando os valores que aparentemente são anomalias em meu gráfico de k-means

df_distante = df_consolidado_json[df_consolidado_json['cluster'] == 0]

# Olhe os valores estatísticos
print(df_distante.describe())
# Verifique datas/horas desses eventos:
print(df_distante[['triggerTs_global', 'peak_time', 'rms_time']])


# Método do Cotovelo (Elbow Method)
# Testando de 1 a 10 clusters
#wss = []
#for k in range(1, 11):
#    kmeans = KMeans(n_clusters=k, random_state=42)
#    kmeans.fit(X)
#    wss.append(kmeans.inertia_)  # inertia_ é o WSS

# Plot do Método do Cotovelo
#plt.figure(figsize=(8, 5))
#plt.plot(range(1, 11), wss, 'bo-')
#plt.xlabel('Número de Clusters')
#plt.ylabel('WSS (Soma das Distâncias Internas)')
#plt.title('Método do Cotovelo')
#plt.grid(True)
#plt.show()



