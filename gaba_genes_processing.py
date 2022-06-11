import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import data_plot_utils
import time
import utils
# import ml_processing
import matplotlib.colors as mcolors
from sklearn import decomposition
from matplotlib import pyplot as plt

def mark_nueronal_genes(path_to_features, path_avg_clust_cells):
    # nueronal_genes_lst = ['GABA','Glut1','Glut2','non-neurons','doublets']
    nueronal_genes_lst = ['Hbb-bs','Plp1','Avp','Hba-a1']
    id_doublets = 4
    features = pd.read_csv(path_to_features, header=0)
    df_avg_clust_cells = pd.read_csv(path_avg_clust_cells, header=0, index_col=0)
    features.set_axis(['num', 'geneID', 'geneName'], axis=1, inplace=True)
    features.set_index('geneName', inplace=True)
    found_genes = []
    not_in_data_gens = []
    for gene_name in nueronal_genes_lst:
        gene_id = int(features.at[gene_name, 'num'])
        if gene_id not in df_avg_clust_cells.index:
            print(f"!!! gen {gene_name} (gen id {gene_id}) is not in the stacked matrix data")
            not_in_data_gens.append((gene_name, gene_id))
            continue
        print(f'gen id for {gene_name} is {gene_id}')
        found_genes.append(gene_id)
    df_nueronal_avg = df_avg_clust_cells.loc[found_genes]
    df_max_nueronal = df_nueronal_avg.idxmax(axis=0) 
    for i in df_nueronal_avg.columns:
        two_largest_genes = df_avg_clust_cells[i].nlargest(2)
        if two_largest_genes.max() < two_largest_genes.min()*2:
            df_max_nueronal[i] = id_doublets
        
    pass

if __name__ == '__main__':
    mark_nueronal_genes(path_to_features='./csv_data2/features.csv', path_avg_clust_cells='./clusttered_data/avg_clust_cells.csv')
    print("Done")