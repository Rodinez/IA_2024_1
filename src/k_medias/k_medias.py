import pandas as pd
import numpy as np
import argparse
import os

def euclidean(u, v):
    u = np.asarray(u)
    v = np.asarray(v)
    return np.sqrt(np.sum(np.square(u - v)))

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('clusters', type=int)
parser.add_argument('iterations', type=int)
args = parser.parse_args()

args_dict = vars(args)
clusters = args_dict['clusters']
iterations = args_dict['iterations']
dataset = args_dict['dataset']

try:
    if clusters <= 0:
        raise ValueError("The parameter 'clusters' must be a numeric value greater than 0")
except ValueError as e:
    print(e)
    exit(1)
try:
    if iterations <= 0:
        raise ValueError("The parameter 'iterations' must be a numeric value greater than 0")
except ValueError as e:
    print(e)
    exit(1)
try:
    if not os.path.exists(f'../dados_e_planilha/datasets/{dataset}'):
        raise FileNotFoundError(f'No such file named {dataset}, it must be a txt file (do not forget the extension)')
except FileNotFoundError as e:
    print(e)
    exit(1) 

dataframe = pd.read_csv(f'../dados_e_planilha/datasets/{dataset}', sep='\t', header=None, names=['Sample', 'A1', 'A2'], skiprows=1)
numSamples = dataframe.shape[0]

try:
    if clusters > numSamples:
        raise ValueError(f'There are more clusters than samples, the number of clusters must be equal or lower than {numSamples}')
except ValueError as e:
    print(e)
    exit(1)

centroids = []
numSamplesPerInitialCentroid = numSamples // clusters

for i in range(clusters):
    meanA1 = 0
    meanA2 = 0
    
    for j in range(numSamplesPerInitialCentroid):
        index = i * numSamplesPerInitialCentroid + j
        meanA1 += dataframe.loc[index]['A1']
        meanA2 += dataframe.loc[index]['A2']
    
    meanA1 /= numSamplesPerInitialCentroid
    meanA2 /= numSamplesPerInitialCentroid
    
    centroidName = 'C'+ f'{i}'
    centroids.append((centroidName, meanA1, meanA2))
    
euclideanDistances = [[] for _ in range(clusters)]
lowerEuclideanDistances = []
    
for i in range(iterations):
    
    for j in range(clusters):
        euclideanDistances[j].clear()
    lowerEuclideanDistances.clear()
    
    for idx, centroid in enumerate(centroids):
        valuesCentroid = (centroid[1], centroid[2])
        for k in range(numSamples):
            valuesSample = (dataframe.loc[k]['A1'], dataframe.loc[k]['A2'])
            distance = euclidean(valuesCentroid, valuesSample)
            euclideanDistances[idx].append(distance)
    
    for j in range(numSamples):
        lowerDistancesToCentroids = min(range(clusters), key=lambda i: euclideanDistances[i][j])
        lowerEuclideanDistances.append(lowerDistancesToCentroids)
    
    centroids.clear()
    for j in range(clusters):
        meanA1 = 0
        meanA2 = 0
        counter = lowerEuclideanDistances.count(j)
        for k in range(numSamples):
            if lowerEuclideanDistances[k] == j:
                meanA1 += dataframe.loc[k]['A1']
                meanA2 += dataframe.loc[k]['A2']
        
        meanA1 /= counter
        meanA2 /= counter
    
        centroidName = 'C'+ f'{j}'
        centroids.append((centroidName, meanA1, meanA2))
        

resultDataframe = pd.DataFrame({
    'sample_label': dataframe['Sample'],
    'centroid_index': lowerEuclideanDistances
})

outputFile = dataset[:-3] + 'clu'
resultDataframe.to_csv(f'../out/k_medias_{outputFile}', sep=' ', index=False, header=False)              