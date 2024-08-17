import pandas as pd
import argparse
from scipy.spatial.distance import euclidean
import os

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