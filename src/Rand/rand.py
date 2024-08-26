import pandas as pd
import argparse
import os
import sklearn.metrics as sl


# essa função itera sobre as linhas do datagrama em bussca do ID passado
#Quando acha o ID, retorna o seu cluster
def nova_coluna(id, dataframeT):
    for id_coluna, linha in dataframeT.iterrows():
        if id in linha['A1']:
            return id_coluna


# essa função simula uma entrada no sistema, usa ela para o Google Colab
datasetReal = 'monkeyReal1.clu'
nivelDendrograma = '5'
algoritmo = 'complete-link'
sys.argv = ['script.py', datasetReal, nivelDendrograma, algoritmo]

parser = argparse.ArgumentParser()
parser.add_argument('datasetReal', type=str)
parser.add_argument('nivel', type=int)
parser.add_argument('algoritmo', type=str)
args = parser.parse_args()

args_dict = vars(args)
k = args_dict['nivel']
datasetR = args_dict['datasetReal']
alg = args_dict['algoritmo']

#monta o nome do arquivo de teste e dos seus objetos
if (datasetR.startswith('monkey')):
  datasetT =  datasetReal[:-9] + f'-nivel_{k}.clu'
  nome = 'monkeyc1g1s'
  offset = 11
elif (datasetR.startswith('c2ds1')):
  nome = 'c2sp1s'
  datasetT =  datasetReal[:-8] + f'-nivel_{k}.clu'
  offset = 6
else:
  nome = 'c2g1s'
  datasetT =  datasetReal[:-8] + f'-nivel_{k}.clu'
  offset = 5

#monta dataframes com as informações
dataframeR = pd.read_csv(
    f'https://raw.githubusercontent.com/Rodinez/IA_2024_1/main/src/dados_e_planilha/datasets/{datasetR}', sep='\t',
    header=None, names=['Sample', 'A1'], skiprows=0)

#para montar o dataframe de teste, faremos de um jeito diferete:
#como nele há os clusters e a lista de objetos pertencentes a cada cluster, transformaremos 
#o cluster em uma coluna e o resto(objetos) em outra.
data = []

if (alg.startswith('k')):
  alg = ''
elif (alg.startswith('single')):
  alg = '2' 
else:
  alg = '3'

with open(f'out{alg}/{datasetT}', 'r') as file:
    for line in file:
        # Remover espaços extras e quebras de linha
        line = line.strip()
        
        # Separar o primeiro elemento (sample) do restante
        elementos = line.split(' ', 1)  # Divide no primeiro espaço encontrado
        sample = elementos[0]
        
        # Verifica se há elementos adicionais
        if len(elementos) > 1:
            # Separa os elementos restantes por vírgula
            valores = elementos[1].split(',')
        else:
            valores = []  # Lista vazia se não houver elementos
        
        # Adiciona os dados processados à lista
        data.append({'Sample': sample, 'A1': valores})

# Cria o DataFrame a partir da lista de dicionários
dataframeT = pd.DataFrame(data)

#Para facilitar a busca por cluster, analisaremos apenas a parte numérica do identificador do objeto
dataframeR['Sample'] = dataframeR['Sample'].str[offset:]

#Criamos uma nova coluna contendo o clusters em que o objeto foi classificado nos testes.

dataframeR['B1'] = dataframeR['Sample'].apply(lambda id: nova_coluna(str((int(id)-1)), dataframeT))

display(dataframeR)

print(f"O índice Rand ajustado é  {sl.adjusted_rand_score(dataframeR['A1'],dataframeR['B1'])}")


