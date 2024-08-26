import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def euclidean(u, v):
    u = np.asarray(u)
    v = np.asarray(v)
    return np.sqrt(np.sum(np.square(u - v)))

def plot_dataset(dataframe, clusters, suptitle=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(dataframe['A1'], dataframe['A2'], s=10, c=colorize(clusters))
    ax.set_title('{0} - {1} clusters'.format('Dataset', len(np.unique(clusters))))
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=16)
    
    plt.show()

def colorize(clusters):
    color_palette = ['#910101', '#036CD7', '#078F8F', '#000000', '#FDFD6F', '#B16FFD', '#22FF25',
                     '#074752', '#8F4A00', '#FE6FB5', '#6BB6FE', '#DF6B00', '#48018D', '#FCB5DA']
    labels = np.unique(clusters)
    if len(labels) > len(color_palette):
        color_palette += ["#%06x" % c for c in np.random.randint(0, 0xFFFFFF, len(labels)-len(color_palette))]
    
    color_array = np.empty(clusters.shape, dtype=object)
    for label, color in zip(labels, color_palette):
        color_array[np.where(clusters == label)] = color
    
    return color_array

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
        
        if counter == 0:
            centroidName = 'C'+ f'{j}'
            centroids.append((centroidName, meanA1, meanA2))
        else:
            meanA1 /= counter
            meanA2 /= counter
        
            centroidName = 'C'+ f'{j}'
            centroids.append((centroidName, meanA1, meanA2))
        

resultDataframe = pd.DataFrame({
    'sample_label': dataframe['Sample'],
    'centroid_index': lowerEuclideanDistances
})

plot_dataset(dataframe, np.array(resultDataframe['centroid_index']), suptitle="k_médias")

outputFile = dataset[:-3] + 'clu'
resultDataframe.to_csv(f'../out/k_medias_{outputFile}', sep=' ', index=False, header=False)              