import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações visuais
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

# Caminho onde estão os arquivos CSV
caminho_pasta_csv = r'C:\Users\mathe\OneDrive\Área de Trabalho\Estágio\Projeto 2\Data csv'

# Lista para armazenar os DataFrames
lista_csv = []

# Lendo todos os CSVs da pasta
for arquivo in glob.glob(os.path.join(caminho_pasta_csv, "*.csv")):
    try:
        df_csv = pd.read_csv(arquivo)
        lista_csv.append(df_csv)
    except Exception as e:
        print(f'Erro ao tentar ler o arquivo: {arquivo} - {e}')

# Verificando se há arquivos válidos
if not lista_csv:
    print("Nenhum arquivo CSV válido foi carregado.")
    exit()

# Concatenando todos os DataFrames
df = pd.concat(lista_csv, ignore_index=True)
print("Consolidação feita com sucesso!")

# Exibindo primeiras linhas
print(df.head(3))

# -----------------------
# PRÉ-PROCESSAMENTO
# -----------------------

# Corrigindo valores nulos ou estranhos
df = df.dropna(subset=['T', 'R', 'V', 'Time'])

# Convertendo a coluna Time para hora do dia (assumindo que Time está em segundos)
df['Hora'] = (df['Time'] // 3600 % 24).astype(int)

# Criando Periodo com base na Hora, se não existir
if 'Periodo' not in df.columns:
    condicoes = [
        (df['Hora'] >= 0) & (df['Hora'] < 6),
        (df['Hora'] >= 6) & (df['Hora'] < 12),
        (df['Hora'] >= 12) & (df['Hora'] < 18),
        (df['Hora'] >= 18)
    ]
    categorias = ['Madrugada', 'Manhã', 'Tarde', 'Noite']
    df['Periodo'] = np.select(condicoes, categorias, default='Indefinido')

df['Periodo'] = pd.Categorical(df['Periodo'], categories=categorias, ordered=True)

# -----------------------
# GRÁFICO DE MÉDIAS HORÁRIAS - COM CORREÇÃO
# -----------------------

# Verificando se há valores suficientes para plottar
if not df[['T', 'R', 'V', 'Hora']].empty:
    medias_por_hora = df.groupby('Hora')[['T', 'R', 'V']].mean()

    plt.figure(figsize=(14, 7))
    cores = ['#1f77b4', '#2ca02c', '#d62728']

    for i, componente in enumerate(['T', 'R', 'V']):
        plt.plot(medias_por_hora.index,
                 medias_por_hora[componente],
                 label=f'Componente {componente}',
                 color=cores[i],
                 linewidth=2)
        
        plt.fill_between(medias_por_hora.index,
                         medias_por_hora[componente],
                         color=cores[i],
                         alpha=0.15)

    # Ajustando escala do eixo Y para garantir visibilidade
    plt.ylim(min(df[['T', 'R', 'V']].min()) * 0.9, max(df[['T', 'R', 'V']].max()) * 1.1)

    plt.title('Variação Horária das Componentes Sísmicas', fontsize=16, fontweight='bold')
    plt.xlabel('Hora do Dia (0–23)', fontsize=12)
    plt.ylabel('Amplitude Média', fontsize=12)
    plt.xticks(np.arange(0, 24, 1))
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("Dados insuficientes para gerar o gráfico de médias horárias.")

# -----------------------
# MATRIZ DE CORRELAÇÃO
# -----------------------

df_corr = df[['T', 'R', 'V', 'Hora']].corr()

plt.figure(figsize=(8, 6))
mask = np.triu(np.ones_like(df_corr, dtype=bool))

sns.heatmap(df_corr,
            annot=True,
            cmap='coolwarm',
            fmt='.2f',
            linewidths=0.5,
            linecolor='white',
            mask=mask,
            annot_kws={"size": 11},
            cbar_kws={'label': 'Coeficiente de Correlação'})

plt.title('Correlação entre Componentes Sísmicas e Hora do Dia', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# -----------------------
# VALIDAÇÃO DA CORRELAÇÃO - DISPERSÃO
# -----------------------

plt.figure(figsize=(12, 5))
sns.pairplot(df[['T', 'R', 'V', 'Hora']], diag_kind='kde', plot_kws={'alpha': 0.5})
plt.suptitle('Distribuição e Relação entre as Variáveis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# -----------------------
# BOXPLOT POR PERÍODO - COMPONENTE T
# -----------------------

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Periodo', y='T', palette='pastel')

plt.title('Distribuição da Componente T por Período do Dia', fontsize=14, fontweight='bold')
plt.xlabel('Período do Dia')
plt.ylabel('Amplitude da Componente T')
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()