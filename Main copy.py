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
import parse as py


# Leitura dos dados csv's consolidação de todos.
caminho_pasta_csv = r'C:\Users\mathe\OneDrive\Área de Trabalho\Estágio\Projeto 2\Data csv'
caminho_pasta_freq = r'C:\Users\mathe\OneDrive\Área de Trabalho\Estágio\Projeto 2\Data freq'

num_linha_freq = [] # Cria uma lista vazia para armazenar os dados
arquivos_csv = glob.glob(os.path.join(caminho_pasta_freq, "*.csv")) # Pega todos os arquivos da minhas pasta com formato .csv

for arquivo in arquivos_csv: # Para cada arquivo na minha pasta faça:
    try: # Tente:
        df_csv = pd.read_csv(arquivo) # Tranforma cada arquivo lido em um dataframe 
        num_linha_freq.append({"ficheiro": os.path.basename(arquivo),"num_linhas": df_csv.shape[0]}) # Adiciona ao final o numero de linhas
    except Exception as e:
        print(f'Erro ao tentar ler esse arquivo aqui: {e}') # Caso ele nao consiga pegar me de um erro e me diga em qual arquivo não conseguiu

if num_linha_freq: # Se o for de cima for bem sucedido 
    df_L_freq = pd.DataFrame(num_linha_freq) # cria df_consolidado  e concatene com a minha lista tranformando em um DataFrame
    print('Consolidação feita com sucesso! DATA: ') 
else:
    print('Aqui deu ruim meu chapa  :( ') # Se esse if não der certo, or algum motivo não conseguir concatenar, me mostre esse print/erro





# Leitura dos dados e consolidação dos dados freq.
#list_freq = [] # Cria uma lista vazia para armazenar os dados.
#arquivos_freq = glob.glob(os.path.join(caminho_pasta_freq, "*.csv")) 

# for arquivo in arquivos_freq:  # Para cada arquivo na minha pasta faça:
#     try: # Tente:
#         df_freq = pd.read_csv(arquivo) # Tranforma cada arquivo lido em um dataframe 
#         list_freq.append(df_freq)  # Adiciona ao final da minha lista vazia.
#     except Exception as e:
#         print(f'Erro ao tentar ler isso aqui meu chapa') # Caso ele nao consiga pegar me de um erro e me diga em qual arquivo não conseguiu

# if num_linha_freq:
#     df_consolidado_freq = pd.concat(list_freq, ignore_index=True)
#     print('Consolidação feita com sucesso! FREQ: ') # cria df_consolidado  e concatene com a minha lista tranformando em um DataFrame
# else:
#     print('Não consegui :( ') # Se esse if não der certo, or algum motivo não conseguir concatenar, me mostre esse print/erro


# # Lista que vai armazenar os dados consolidados dos arquivos JSON 
# lista_dfs = []






# Caminho da pasta com os JSONs
caminho_pasta = "Data Json"

# arquivos_json = glob.glob(os.path.join(caminho_pasta, "*.json"))

# # Loop por todos os arquivos .json da pasta
# for file in glob.glob(os.path.join(caminho_pasta, "*.json")):
#     with open(file, "r", encoding="utf-8") as f:
#         dados = json.load(f)

#         for recorder_id, event_data in dados['eventFiles'].items():
#             # Informações principais do evento e da estação.
#             base_info = {
#                 'id_global': dados.get('id'), # Pega a coluna de ID.
#                 'triggerTs_global': dados.get('triggerTs'), # Pega a coluna de Triggers
#                 'data_arquivo': os.path.basename(file)  # nome do arquivo json/ apenas para rastreio
#             }

#             # Acessa os dados por canal (ex: canais T, R, V) dentro da chave 'cf'
#             if 'df' in event_data and 'cf' in event_data['df']:
#                 # Para cada canal (T, R, V) nos dados temporais
#                 for canal in event_data['df']['cf']:
#                     # Cria uma nova linha com as informações básicas
#                     linha = base_info.copy()
                    
#                     # Adiciona os dados do domínio do tempo
#                     linha['canal'] = canal['chName']
#                     linha['peak_time'] = canal['peak']
#                     linha['rms_time'] = canal['rms']
#                     linha['value_time'] = canal['value']
                    
#                     # Inicializa as colunas de frequência como NaN
#                     #for i in range(5):
#                        # linha[f'freq_{i}'] = np.nan # Criado apenas porque o Pandas exige que na criação as linhsa tenham as mesmas colunas, depois será preenchido
                        
#                     # Adiciona os dados de frequência existirem
#                     if 'dfFft' in event_data and 'peak' in event_data['dfFft']:
#                         freq_data = next((item for item in event_data['dfFft']['peak']  # Encontra os dados de frequência para este canal específico
#                                         if item['chName'] == canal['chName']), None)
                        
#                         if freq_data:
#                             # Pega os 5 principais picos de frequência
#                             for i, pico in enumerate(freq_data['value'][:5]):
#                                 linha[f'freq_{i}'] = pico['freq']
#                                 linha[f'ampl_{i}'] = pico['ampl']
                    
#                     # Adiciona a linha completa à lista
#                     lista_dfs.append(linha)

# # Cria o DataFrame consolidado
# df_consolidado_json = pd.DataFrame(lista_dfs)

df_consolidado_json = py.parse_all_events(caminho_pasta)

# Print para confirmar como os nossos dados estão sendo armazenados
# print(df_consolidado_json.shape)
# print(df_L_freq.shape)
# print(df_L_freq.shape)
# colunas_analise = df_consolidado_json.loc[:, ['canal'] + list(df_consolidado_json.loc[:, 'peak_time':].columns)]
# print(colunas_analise.head())  # Print de confirmação de como está os dados

#Limpeza Pré processamento de dados para cluster
#padrões = ['triggerTs', 'cf_value']
padrões = ['fft_freq']
df_filtrado = df_consolidado_json[[col for col in df_consolidado_json.columns if any(p in col for p in padrões)]]
#df_cluster = py.convert_triggerTs_to_float(df_filtrado) # Criando uma cópia e trabaçhando com ela.
df_cluster = df_filtrado

#df_cluster.fillna(df_cluster.median(numeric_only=True), inplace=True) # Removedno valores nulos e  preenchendo com a mediana

# Normalizar dados para o k-Means.
scaler = StandardScaler()
X = scaler.fit_transform(df_cluster) # Aplica a padronização nos dados, E depois transforma os dados (transform), O resultado X contém os dados padronizados

# A criação com 3 cluster foi feita após uma análise do código abaixo (Elbow Method)
kmeans = KMeans(n_clusters=3, random_state=42)
df_consolidado_json['cluster'] = kmeans.fit_predict(X)
print(f'valor da variavel X: {X}')


#------------Descobrindo quantidade de cluster----------
# Método do Cotovelo (Elbow Method)

wss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wss.append(kmeans.inertia_)  # inertia_ é o WSS

#Plot do Método do Cotovelo
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wss, 'bo-')
plt.xlabel('Número de Clusters')
plt.ylabel('WSS (Soma das Distâncias Internas)')
plt.title('Método do Cotovelo')
plt.grid(True)
plt.show()
#------------------------------------------------------

# Fazendo PCA e Plotando. (Principal Component Analysis - Análise de Componentes Principais)
pca = PCA(n_components=2) # Definindo numero de componentes principais.
X_pca = pca.fit_transform(X) # ajustando modelo PCA

plt.figure(figsize=(10, 6))  
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df_consolidado_json['cluster'], cmap='viridis')
plt.title('Distribuição de Eventos Sísmicos com Base em Padrões de Frequência e Intensidade')
plt.xlabel('Padrão de Intensidade Sísmica')
plt.ylabel('Variação de Frequência / Amplitude')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()
loadings = pd.DataFrame(pca.components_.T,
                        index=df_consolidado_json.columns[:-1],
                        columns=['PC1', 'PC2'])

print("Loadings:")
print(loadings)



# Conversão de data e hora dde colunas triggerTs_global
#df_consolidado_json['triggerTs_global'] = pd.to_datetime(df_consolidado_json['triggerTs_global'])

# Obtém um mapa de cores pré-definido ('viridis') para usar na visualização
cmap = plt.get_cmap('viridis')
# Mapeia cada valor de 'cluster' para uma cor do mapa de cores (normalizado pela quantidade de clusters únicos)
cores = df_consolidado_json['cluster'].map(lambda x: cmap(x / df_consolidado_json['cluster'].nunique()))
# Cria uma nova coluna 'hora' extraindo apenas a hora do timestamp 'triggerTs_global'
df_consolidado_json['hora'] = pd.to_datetime(df_consolidado_json['triggerTs']).dt.hour

plt.figure(figsize=(10, 5)) 
for cluster in sorted(df_consolidado_json['cluster'].unique()): # Loop que itera sobre cada valor único de 'cluster', em ordem crescente
    dados_cluster = df_consolidado_json[df_consolidado_json['cluster'] == cluster]  # Filtra o DataFrame para pegar apenas os dados do cluster atual
    plt.hist(dados_cluster['hora'], bins=24, alpha=0.6, label=f'Cluster {cluster}', color=cmap(cluster / df_consolidado_json['cluster'].nunique()))

plt.title('Distribuição de Eventos por Hora do Dia (por Cluster)')
plt.xlabel('Hora do Dia')
plt.ylabel('Número de Eventos')
plt.legend()
plt.grid(True)
plt.show()

# Print para verificarmos as variáveis dos eventos do cluster.
#print(f'Dados para visão de variável colunas de cluster: \n {colunas_para_cluster}')

# Identificando os valores que aparentemente são anomalias em meu gráfico de k-means
df_distante = df_consolidado_json[df_consolidado_json['cluster'] == 0]
print(df_distante.describe())
print(df_distante[['triggerTs_global', 'peak_time', 'rms_time']]) # Verifique datas/horas desses eventos



# =============================================
# ANÁLISE DE ANOMALIAS EM CLUSTERS - VERSÃO FUNCIONAL

# 1. Primeiro identifique qual cluster está realmente isolado (provavelmente o cluster 2)
cluster_anomalia = 2  # Altere para o número do cluster que aparece isolado no seu gráfico

# 2. Transformação dos centróides para o espaço PCA
centroides_pca = pca.transform(kmeans.cluster_centers_)

# 3. Calcular distâncias para o cluster suspeito
df_anomalias = df_consolidado_json[df_consolidado_json['cluster'] == cluster_anomalia].copy()
distancias = pairwise_distances(
    X_pca[df_consolidado_json['cluster'] == cluster_anomalia],
    [centroides_pca[cluster_anomalia]]
)
df_anomalias['distancia_centroide'] = distancias

# 4. Definir limiar para anomalias (2.5 desvios padrão da média)
limiar = df_anomalias['distancia_centroide'].mean() + 2.5 * df_anomalias['distancia_centroide'].std()
anomalias_reais = df_anomalias[df_anomalias['distancia_centroide'] > limiar]

# 5. Visualização corrigida
plt.figure(figsize=(12, 7))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=df_consolidado_json['cluster'], alpha=0.5, cmap='viridis')

# Destacar apenas as anomalias reais
plt.scatter(X_pca[anomalias_reais.index,0], X_pca[anomalias_reais.index,1], 
            edgecolors='red', facecolors='none', s=150, linewidths=2, label='Anomalias reais')

# Adicionar centróides
plt.scatter(centroides_pca[:,0], centroides_pca[:,1], marker='X', s=200, c='black', label='Centróides')

# Adicionar informações de data para as anomalias
for idx, row in anomalias_reais.iterrows():
    data_formatada = row['triggerTs_global'].strftime('%Y-%m-%d %H:%M')
    plt.annotate(data_formatada, 
                 (X_pca[idx,0], X_pca[idx,1]),
                 textcoords="offset points",
                 xytext=(0,10),
                 ha='center',
                 fontsize=8,
                 color='red')

plt.title('Visualização do PCA (Análise de Componentes Principais)\nCom Anomalias Destacadas')
plt.xlabel('Padrão de Intensidade Sísmica')
plt.ylabel('Variação de Frequência / Amplitude ')
plt.legend()
plt.grid(True)
plt.show()

# 6. Imprimir informações detalhadas das anomalias
print("\nINFORMAÇÕES DAS ANOMALIAS IDENTIFICADAS:")
print(anomalias_reais[['triggerTs_global', 'peak_time', 'rms_time', 'distancia_centroide']].sort_values('distancia_centroide', ascending=False))

# 7. Análise temporal das anomalias
print("\n DISTRIBUIÇÃO TEMPORAL DAS ANOMALIAS:")
anomalias_por_dia = anomalias_reais['triggerTs_global'].dt.date.value_counts().sort_index()
print(anomalias_por_dia)

plt.figure(figsize=(10,4))
anomalias_por_dia.plot(kind='bar', color='black', title='Distribuição das Anomalias por Dia')
plt.xlabel('Data')
plt.ylabel('Número de Anomalias')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

