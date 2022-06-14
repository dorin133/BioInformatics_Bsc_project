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
from distinctipy import distinctipy


def clustter_nueronal_genes(path_to_features, path_frac_clust_cells, path_in_cluseters, path_out):
    utils.write_log(f"starting clustter_nueronal_genes: nueral gene expression on T-SNE's scatter plot")
    # nueronal_genes_lst = ['GABA','Glut1','Glut2','non-neurons','doublets']
    nueronal_genes_lst = ['Gad2','Slc17a7','Slc17a6']
    gaba_id = 0
    glut1_id = 1
    glut2_id = 2
    id_doublets = 3
    id_no_type = 4
    marker_map = {
        0: 'Gaba',
        1: 'Glut1',
        2: 'Glut2',
        3: 'Doublets',
        4: 'No Type'
    }
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
    # df_nueronal_frac = df_nueronal_frac_T.T
    # if df_nueronal_frac_T['max_expression'] < df_nueronal_frac_T['largest2'] * 2:
    # df_nueronal_frac_T['id_max_expression'] = df_nueronal_frac_T['id_max_expression'] if (df_nueronal_frac_T['max_expression'] < df_nueronal_frac_T['largest2'] * 2) else id_doublets
    clust_nueral_class = {}
    df_nueronal_frac.index.name = 'gene_id'
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
    
    # df_nueronal_frac_T = df_nueronal_frac.T
    # df_nueronal_frac_T['nueral_idx'] = list(clust_nueral_class.values())
    # df_nueronal_frac_T['nueral_labels'] = df_nueronal_frac_T['nueral_idx'].map(marker_map)
    # print("!")

    map_nueral_class = {float(k): marker_map[v] for k, v in clust_nueral_class.items()}
    map_nueral_class[-1.0] = 'No Type'
    clusters_df = pd.read_csv(path_in_cluseters, header=0, index_col=0).T
    clusters_df['nueral_labels'] = clusters_df['linkage_labels'].map(map_nueral_class)
    clusters_df = clusters_df.T
    clusters_df.to_csv(path_out, sep=',')
    utils.write_log(f"finished clustter_nueronal_genes")


def plot_nueral_gene_expression(path_clust_tsne_data='./clusttered_data/clust_tsne_data.csv', plots_folder='./plots_folder1/testing2_out'):
    utils.write_log(f"starting plot_nueral_gene_expression")
    df_tsne = pd.read_csv(path_clust_tsne_data, index_col=0, header=0).T
    df_tsne['tsne-2d-one'] = df_tsne['tsne-2d-one'].astype('float64')
    df_tsne['tsne-2d-two'] = df_tsne['tsne-2d-two'].astype('float64')
    sns.scatterplot(
            x = 'tsne-2d-one',
            y = 'tsne-2d-two',
            hue = 'nueral_labels',
            data = df_tsne,
            palette=distinctipy.get_colors(len(df_tsne['nueral_labels'].unique()), pastel_factor=0.5),
            legend="full",
            alpha = 0.3
        )
    plt.title(f"Nueral Gene Experession Scatter")
    name_path = "nueral_gene_expression"
    data_plot_utils.save_plots(plt, f'{plots_folder}/{name_path}')
    plt.show()


def avg_and_fraction_clustter_expression(path_in_stack, path_tsne_dbscan_data, path_out_avg_clust_cell, path_out_frac):
        utils.write_log(f"starting avg_and_fraction_clustter_expression")
        df_stack = pd.read_csv(path_in_stack, index_col=0, header=0)
        df_tsne_dbscan = pd.read_csv(path_tsne_dbscan_data, index_col=0, header=0)
        # df = pd.concat([df_stack, df_dbscan])
        
        df_stack_log = np.log2(df_stack+1)
        df_stack_normalized = df_stack_log.sub(df_stack_log.mean(axis=1), axis=0)
        df_stack_normalized = df_stack_normalized.div(df_stack_log.std(axis=1), axis=0)  # TODO verify this
        # add to the raw gene expression data (already normalized) the clustter idx of each cell
        df_stack_normalized.loc['linkage_labels'] = df_tsne_dbscan.iloc[2]
        df_stack_log.loc['linkage_labels'] = df_tsne_dbscan.iloc[2]
        # the mean() next line is taken on each clustter seperately
        df_stack_normalized_avg = (df_stack_normalized.T.groupby(by = 'linkage_labels', sort=True).mean()).iloc[1:] # iloc[1:] to drop the '-1' clustter idx 
        # for all cells in each clustter - sum the gene expression value for every gene 
        df_stack_frac = (df_stack_log.T.groupby(by = 'linkage_labels', sort=True).sum()).iloc[1:] # iloc[1:] to drop the '-1' clustter idx 
        # now, for each row (meaning, for each clustter) divide by the sum of gene expression of all genes
        # each gene now (meaning every value in the metrics) is a fraction of how much he is expressed 
        df_stack_frac = df_stack_frac.div(df_stack_frac.sum(axis=1), axis=0) 
        # make sure all values in sanity_checker are one - DONE
        sanity_checker = df_stack_frac.sum(axis=1)
        utils.write_log(f"saving a table of each clustter's its average cell and average expression percentage of each gene")
        df_stack_normalized_avg.T.to_csv(path_out_avg_clust_cell, sep=',')
        df_stack_frac.T.to_csv(path_out_frac, sep=',')
        utils.write_log(f"finished avg_and_fraction_clustter_expression, results saved to: {path_out_avg_clust_cell} and {path_out_frac}")


def translate_clustter_data(path_in_clustter_data, nueral_idx_arr, path_out_clustter_data ='./clusttered_data/clust_idx_translation_table.csv'):
        utils.write_log(f"starting translate_clustter_data : adding nueral gene clustter index")
        # df_translation = pd.read_csv(path_in_translation, index_col=0, header=0)
        df_clust_data = pd.read_csv(path_in_clustter_data, index_col=0, header=0)
        # initialization of the new row
        # we'll soon replace the dbscan_labels values in this row
        df_clust_data.loc['nueral_labels'] = 4
        for i in df_clust_data.columns:
            col_in_translation_table = int(df_clust_data[i]['linkage_labels'])
            if not col_in_translation_table == -1:
                df_clust_data[i]['nueral_labels'] = nueral_idx_arr[col_in_translation_table]
        df_clust_data.to_csv(path_out_clustter_data, sep=',')
        utils.write_log(f"finished translate_clustter_data: translation table of nueral gene clustter indeces saved to {path_out_clustter_data}")


if __name__ == '__main__':
    # avg_and_fraction_clustter_expression(path_in_stack='./merged_data5/stacked_1.csv',
    #                           path_tsne_dbscan_data='./clusttered_data/clust_tsne_data.csv', path_out_avg_clust_cell = './clusttered_data/avg_clust_cells_stk1.csv',
    #                           path_out_frac='./clusttered_data/frac_clust_cells_stk1.csv')
    # arr = clustter_nueronal_genes(path_to_features='./csv_data2/features.csv',
    #                               path_frac_clust_cells='./clusttered_data/frac_clust_cells_stk1.csv',
    #                               path_in_cluseters='./clusttered_data/clust_tsne_data.csv',
    #                               path_out='./clusttered_data/tsne_and_clust_labels.csv')
    plot_nueral_gene_expression(path_clust_tsne_data='./clusttered_data/tsne_and_clust_labels.csv', plots_folder='./plots_folder1/testing2_out')
    print("Done")