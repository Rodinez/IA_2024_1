import numpy as np
import pandas as pd
import argparse
import scipy.spatial as sp
import os
import sys


def atualiza_distancias(dataframe, distancias, juntar_cluster):
    # Atualiza as distâncias do novo cluster (juntar_cluster[0]) com todos os outros clusters
    distancias[juntar_cluster[0], :] = np.minimum(distancias[juntar_cluster[0], :], distancias[juntar_cluster[1], :])
    distancias[:, juntar_cluster[0]] = distancias[juntar_cluster[0], :]
    np.fill_diagonal(distancias, np.inf)
    return (distancias)


# essa função simula uma entrada no sistema, usa ela para o Google Colab
sys.argv = ['tst11', 'c2ds1-2sp.txt', '1']

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('clusters', type=int)
args = parser.parse_args()

args_dict = vars(args)
clusters = args_dict['clusters']
dataset = args_dict['dataset']

try:
    if clusters <= 0:
        raise ValueError("The parameter 'clusters' must be a numeric value greater than 0")
except ValueError as e:
    print(e)
    exit(1)
try:
    if not os.path.exists(f'../dados_e_planilha/datasets/{dataset}'):
        raise FileNotFoundError(f'No such file named {dataset}, it must be a txt file (do not forget the extension)')
except FileNotFoundError as e:
    print(e)
    exit(1)

# para testar no google colab
dataframe = pd.read_csv(
    f'https://raw.githubusercontent.com/Rodinez/IA_2024_1/main/src/dados_e_planilha/datasets/{dataset}', sep='\t',
    header=None, names=['Sample', 'A1', 'A2'], skiprows=1)

# para testar no próprio pc
# dataframe = pd.read_csv(f'../dados_e_planilha/datasets/{dataset}', sep='\t', header=None, names=['Sample', 'A1', 'A2'], skiprows=1)
numSamples = dataframe.shape[0]

try:
    if clusters > numSamples:
        raise ValueError(
            f'There are more clusters than samples, the number of clusters must be equal or lower than {numSamples}')
except ValueError as e:
    print(e)
    exit(1)

numero_clusters_min = args.clusters  # este numero varia dentro de um espectro

# converte o dataframe para um vetor de pontos 2D numpy
lista_pontos_a = dataframe[['A1', 'A2']]  # pega os pontos das colunas A1 e A2 apenas
lista_pontos = lista_pontos_a.to_numpy()

# distancias eh uma matriz 2D com a distancia entre todos os pares de pontos
distancias = sp.distance.squareform(sp.distance.pdist(lista_pontos))
# preenche a diagonal de distancias com um numero infinito e depois calcula os minimos da matriz
np.fill_diagonal(distancias, np.inf)

# clusters eh um vetor que cada celula eh uma lista com um ponto da base de dados
# cluster = [[dado1], [dado2], ..., [dadoN]]
clusters = [[i] for i in range(len(lista_pontos))]

# cria uma matriz 3D. Cada matriz 2D dentro dessa matriz 3D é um estado do dendograma (ou seja, contem as combinações entre clusters)
outputFile_list = []
dataframe_list = []
# a primeira matriz tem todos os cluster (primeria partição)
resultDataframe = pd.DataFrame({
    'sample_label': dataframe['Sample'],
    'centroid_index': pd.Series(clusters).apply(lambda x: ','.join(map(str, x)))
})
dataframe_list.append(resultDataframe)
outputFile = dataset[:-3] + 'clu'
outputFile_list.append(outputFile)
# pega o min de cada linha e coloca em um vetor a coluna do minimo dessa linha
matriz_particoes = []
while len(clusters) > numero_clusters_min:
    dist_min = np.inf
    juntar_cluster = (0, 0)

    #pega a coluna do menor elemento de cada linha
    menor_indice = np.array([np.argmin(distancias[i]) for i in range(len(distancias))])
    for i in range(distancias.shape[0]):
        #pega a distancia menor distancia de cada linha e compara qual dela é menor
        dist_min_lc = distancias[i][menor_indice[i]]
        if dist_min_lc < dist_min:
            dist_min = dist_min_lc
            juntar_cluster = (i, menor_indice[i])

    #junta os clusters
    clusters[juntar_cluster[0]].extend(clusters[juntar_cluster[1]])
    #atualiza os menores valores
    distancias = atualiza_distancias(dataframe, distancias, juntar_cluster)
    #remove o cluster que foi juntado
    del clusters[juntar_cluster[1]]
    #remove a distancia que acabamos de encontrar
    distancias = np.delete(distancias, juntar_cluster[1], 1)
    distancias = np.delete(distancias, juntar_cluster[1], 0)
    #coloca os clusters dentro de um vetor de clusters
    matriz_particoes.append([list(cluster) for cluster in clusters])

# Converte as partições em DataFrames e salva-os em uma lista
for nivel, particao in enumerate(matriz_particoes):
    resultDataframe = pd.DataFrame({
        'sample_label': dataframe['Sample'],
        'centroid_index': pd.Series(particao).apply(lambda x: ','.join(map(str, x)))
    })
    dataframe_list.append(resultDataframe)

# Salva cada DataFrame como um arquivo CSV
for i, df in enumerate(dataframe_list):
    outputFile = dataset[:-3] + f'-nivel_{i}.clu'
    df.to_csv(f'../out/{outputFile}', sep=' ', index=False, header=False)
    outputFile_list.append(outputFile)
