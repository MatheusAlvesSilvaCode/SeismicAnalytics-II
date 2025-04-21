import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

caminho_pasta_csv = r'C:\Users\mathe\OneDrive\Área de Trabalho\Estágio\Projeto 2\Data csv'

lista_csv = []
arquivos_csv = glob.glob(os.path.join(caminho_pasta_csv, "*.csv"))

for arquivo in arquivos_csv:
    try:
        df_csv = pd.read_csv(arquivo)
        lista_csv.append(df_csv)
    except Exception as e:
        print(f'Erro ao ler arquivo {arquivo}: {e}')

if lista_csv:
    df_consolidado_csv = pd.concat(lista_csv, ignore_index=True)
    print('Consolidação feita com sucesso! DATA:')
else:
    print('Nenhum arquivo válido encontrado')

print(df_consolidado_csv.head(3))

# Verificação crítica dos dados de tempo
tempo = df_consolidado_csv['Time'].values
print("\nAnálise dos dados temporais:")
print(f"Primeiros valores de tempo: {tempo[:10]}")
print(f"Diferenças temporais: {np.diff(tempo)[:10]}")

# CORREÇÃO DO PROBLEMA: Seus dados provavelmente estão em segundos, mas o intervalo está em milissegundos
dt_correto = 0.0025  # Valor FIXO baseado nos seus dados (2.5ms)
fs_correto = 1 / dt_correto  # 400 Hz (valor típico para dados sísmicos)

# Configuração dos gráficos
plt.figure(figsize=(14, 10))
plt.suptitle('Análise Sísmica Corrigida', fontsize=16)

for i, comp in enumerate(['T', 'R', 'V'], 1):
    sinal = df_consolidado_csv[comp].values
    
    # Downsampling para melhor visualização
    passo = 100  # Mostrar 1 a cada 100 pontos
    tempo_down = tempo[::passo]
    sinal_down = sinal[::passo]
    
    # Gráfico temporal (subplot esquerdo)
    plt.subplot(3, 2, 2*i-1)
    plt.plot(tempo_down, sinal_down, 'k-', linewidth=0.5)  # 'k-' para linha preta
    plt.title(f'Componente {comp} - Tempo (downsampled 1:{passo})')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Cálculo da FFT CORRIGIDO
    n = len(sinal)
    fft_result = np.fft.fft(sinal)
    fft_magnitude = 2 * np.abs(fft_result[:n//2]) / n
    freqs = np.fft.fftfreq(n, d=dt_correto)[:n//2]
    
    # Gráfico de frequência (subplot direito)
    plt.subplot(3, 2, 2*i)
    plt.plot(freqs, fft_magnitude, 'k-', linewidth=0.5)  # 'k-' para linha preta
    plt.title(f'Espectro {comp} (0-{fs_correto/2:.0f} Hz)')
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(0, 50)  # Foco nas baixas frequências
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()