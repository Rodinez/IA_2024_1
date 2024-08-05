import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('clusters', type=int)
parser.add_argument('iterations', type=int)
args = parser.parse_args()

if args.clusters <= 0:
    raise ValueError("The parameter 'clusters' must be a numeric value greater than 0")
if args.iterations <= 0:
    raise ValueError("The parameter 'iterations' must be a numeric value greater than 0")

dataframe = pd.read_csv('../dados_e_planilha/datasets/c2ds1-2sp.txt', sep='\t', header=None, names=['sample_label', 'd1', 'd2'], skiprows=1)

print(dataframe.head(1000))