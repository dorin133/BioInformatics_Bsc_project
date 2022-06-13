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

def mark_nueronal_genes(path_to_features, path_frac_clust_cells):
    # nueronal_genes_lst = ['GABA','Glut1','Glut2','non-neurons','doublets']
    nueronal_genes_lst = ['Gad2','Slc17a7','Slc17a6']
    gaba_id = 0
    glut1_id = 1
    glut2_id = 2
    id_doublets = 3
    id_no_type = 4
    features = pd.read_csv(path_to_features, header=0)
    df_frac_clust_cells = pd.read_csv(path_frac_clust_cells, header=0, index_col=0)
    features.set_axis(['num', 'geneID', 'geneName'], axis=1, inplace=True)
    features.set_index('geneName', inplace=True)
    found_genes = []
    not_in_data_gens = []
    for gene_name in nueronal_genes_lst:
        gene_id = int(features.at[gene_name, 'num'])
        if gene_id not in df_frac_clust_cells.index:
            print(f"!!! gen {gene_name} (gen id {gene_id}) is not in the stacked matrix data")
            not_in_data_gens.append((gene_name, gene_id))
            continue
        print(f'gen id for {gene_name} is {gene_id}')
        found_genes.append(gene_id)
    df_nueronal_frac = (df_frac_clust_cells.loc[found_genes])
    # df_nueronal_frac_T['largest2'] = df_nueronal_frac_T.apply(lambda x: x.nlargest(2).iloc[1], axis=1)
    # df_nueronal_frac_T['max_expression'] = df_nueronal_frac_T.max(axis=1)
    # df_nueronal_frac_T['id_max_expression'] = df_nueronal_frac_T.idxmax(axis=1)
    # df_nueronal_frac = df_nueronal_frac_T.T
    # if df_nueronal_frac_T['max_expression'] < df_nueronal_frac_T['largest2'] * 2:
    # df_nueronal_frac_T['id_max_expression'] = df_nueronal_frac_T['id_max_expression'] if (df_nueronal_frac_T['max_expression'] < df_nueronal_frac_T['largest2'] * 2) else id_doublets
    clust_nueral_class = {}
    for col in df_nueronal_frac.columns:
        nueral_frac_arr = df_nueronal_frac[col].to_numpy()  
        two_largest_idx = (df_nueronal_frac[col].nlargest(2).index).tolist()
        if [found_genes[glut1_id], found_genes[glut2_id]] == two_largest_idx:
            clust_nueral_class[col] = nueral_frac_arr.argmax()
            continue
        # search for doublets:
        if nueral_frac_arr.max() < nueral_frac_arr.min() * 2:
            clust_nueral_class[col] = id_doublets
            continue
        # search for no type 
        if nueral_frac_arr.max() == 0:
            clust_nueral_class[col] = id_no_type

        clust_nueral_class[col] = nueral_frac_arr.argmax()
    print("!")
    pass



        # two_largest_genes = df_frac_clust_cells[i].nlargest(2)
        # if two_largest_genes.max() < two_largest_genes.min()*2 and two_largest_genes not in [1,2]:
        #     df_max_nueronal[i] = id_doublets
        # if two_largest_genes.max() == 0:
        #     df_max_nueronal[i] = id_no_type


        
    pass

if __name__ == '__main__':
    mark_nueronal_genes(path_to_features='./csv_data2/features.csv', path_frac_clust_cells='./clusttered_data/frac_clust_cells.csv')
    print("Done")