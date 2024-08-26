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

dataframe = pd.read_csv(f'../dados_e_planilha/datasets/{dataset}', sep='\t', header=None, names=['Sample', 'A1', 'A2'], skiprows=1)
numSamples = dataframe.shape[0]

try:
    if clusters > numSamples:
        raise ValueError(f'There are more clusters than samples, the number of clusters must be equal or lower than {numSamples}')
except ValueError as e:
    print(e)
    exit(1)
    
euclideanDistances = [[] for _ in range(numSamples)]
partitions = [[] for _ in range(numSamples)]

for i in range(numSamples):
    valuesSample = (dataframe.loc[i]['A1'], dataframe.loc[i]['A2'])
    nameSample = (dataframe.loc[i]['Sample'])
    partitions[0].append(nameSample)
    for j in range(numSamples):
        valuesComparativeSample = (dataframe.loc[j]['A1'], dataframe.loc[j]['A2'])
        distance = euclidean(valuesSample, valuesComparativeSample)
        euclideanDistances[i].append(distance)

partition = 0
while(len(euclideanDistances) != 1):
    partition += 1
    mininumDistance = (9999999, -1, -1)
    for i in range(len(euclideanDistances)):
        for j in range(i + 1, len(euclideanDistances)):
            if mininumDistance[0] > euclideanDistances[i][j]:
                mininumDistance = (euclideanDistances[i][j], i, j)

    newRow = []
    for i in range(len(euclideanDistances)):
        if i == mininumDistance[1]:
            newRow.append(0)
        elif i == mininumDistance[2]:
            continue
        else:
            newRow.append(min(euclideanDistances[mininumDistance[1]][i], euclideanDistances[mininumDistance[2]][i]))
    
    for i in range(len(euclideanDistances)):
        if i == mininumDistance[1]:
            nameSample = partitions[partition - 1][i]
            nameSample2 = partitions[partition - 1][mininumDistance[2]]
            if isinstance(nameSample, tuple):
                new_tuple = nameSample + (nameSample2,) if not isinstance(nameSample2, tuple) else nameSample + nameSample2
            else:
                new_tuple = (nameSample, nameSample2) if not isinstance(nameSample2, tuple) else (nameSample,) + nameSample2
            partitions[partition].append(new_tuple)
        elif i == mininumDistance[2]:
            continue
        else:
            nameSample = partitions[partition - 1][i]
            partitions[partition].append(nameSample)
    
    euclideanDistances.pop(mininumDistance[2])
    euclideanDistances.pop(mininumDistance[1])
    euclideanDistances.insert(mininumDistance[1], newRow)
    for i in range(len(euclideanDistances)):
        euclideanDistances[i][mininumDistance[1]] = newRow[i]
        if i != mininumDistance[1]:
            euclideanDistances[i].pop(mininumDistance[2])


links = []
partition = len(partitions) - clusters
for idx, elem in enumerate(partitions[partition]):
    if isinstance(elem, tuple):
        links.extend([idx] * len(elem))
    else:
        links.append(idx)
       
resultDataframe = pd.DataFrame({
    'sample_label': dataframe['Sample'],
    'cluster': links
})

plot_dataset(dataframe, np.array(resultDataframe['cluster']), suptitle="single_link")

outputFile = dataset[:-3] + 'clu'
resultDataframe.to_csv(f'../out/single_link_{outputFile}', sep=' ', index=False, header=False)  