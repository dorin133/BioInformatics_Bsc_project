from turtle import distance
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import data_plot_utils
import time
import utils
import ml_processing
import matplotlib.colors as mcolors
from sklearn import decomposition
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from bioinfokit import analys
import holoviews as hv 
import hvplot.pandas # use hvplot directly with pandas
hv.extension('bokeh')  # to generate interactive plots 


def sanity_checks(path_in_stack, path_in_dbscan, path_to_features_csv, gene_list, plots_folder):
    utils.write_log(f"starting sanity check: gene expression volume on T-SNE's scatter plot")
    df_stack = pd.read_csv(path_in_stack, index_col=0, header=0)
    # print(df_stack.isna().sum(axis=1))  # TODO not sure why is that
    # df_stack.fillna(0, inplace=True)
    # print(df_stack.isna().sum(axis=1))

    df_dbscan = pd.read_csv(path_in_dbscan, index_col=0, header=0)
    # df = pd.concat([df_stack, df_dbscan])
    # df = df.T

    features = pd.read_csv(path_to_features_csv, header=0)
    features.set_axis(['num', 'geneID', 'geneName'], axis=1, inplace=True)
    features.set_index('geneName', inplace=True)
    relevant_features = features.loc[gene_list].num
    relevant_gene_expressions = df_stack.loc[relevant_features]
    relevant_gene_expressions_log = np.log2(relevant_gene_expressions+1)
    relevant_gene_expressions = relevant_gene_expressions_log.sub(relevant_gene_expressions_log.mean(axis=1), axis=0)
    relevant_gene_expressions = relevant_gene_expressions.div(relevant_gene_expressions_log.std(axis=1), axis=0) # TODO verify this
    x_coor = df_dbscan.T['tsne-2d-one']
    y_coor = df_dbscan.T['tsne-2d-two']
    gene_expressions_max = relevant_gene_expressions.max(axis=1)
    gene_expressions_min = relevant_gene_expressions.min(axis=1)
    # relevant_gene_expressions['is_intresting'] = relevant_gene_expressions.apply(lambda x: x[gene_id] != 0, axis=1)
    for i, gene_name in enumerate(gene_list):

        gene_id = int(features.at[gene_name, 'num'])
        vmin = gene_expressions_min[gene_id]
        vmax = gene_expressions_max[gene_id]
        normalize_val = mcolors.TwoSlopeNorm(vcenter=0, vmin=vmin, vmax=vmax)
        # if gene_id not in relevant_gene_expressions.index:
        #     print(f"!!! gen {gene_name} (gen id {gene_id}) is not in the stacked matrix data")
        #     continue
        # print(f'gen id for {gene_name} is {gene_id}')
        
        plt.figure(figsize=(9, 9))
        sns.scatterplot(
            x=x_coor,
            y=y_coor,
            c=relevant_gene_expressions.loc[gene_id].to_numpy(),
            cmap = cm.RdBu,
            data=relevant_gene_expressions,
            norm = normalize_val,
            # palette='RdBu',
            # legend="auto",
            alpha=0.3
        )
        plt.title(f"Gene Experession Scatter for gene {gene_name} (gene id {gene_id})")

        name_path = "gene_expression_of_"+gene_name
        data_plot_utils.save_plots(plt, f'{plots_folder}/{name_path}')
        # plt.show()
    utils.write_log(f"finished sanity check: plots saved in {plots_folder}")

def linkage_data_prep():
    def avg_and_fraction_clustter_expression(path_in_stack, path_tsne_dbscan_data, path_out_avg_clust_cell, path_out_frac):
        utils.write_log(f"starting avg_and_fraction_clustter_expression")
        df_stack = pd.read_csv(path_in_stack, index_col=0, header=0)
        df_tsne_dbscan = pd.read_csv(path_tsne_dbscan_data, index_col=0, header=0)
        # df = pd.concat([df_stack, df_dbscan])
        
        df_stack_log = np.log2(df_stack+1)
        df_stack_normalized = df_stack_log.sub(df_stack_log.mean(axis=1), axis=0)
        df_stack_normalized = df_stack_normalized.div(df_stack_log.std(axis=1), axis=0) # TODO verify this
        # add to the raw gene expression data (already normalized) the clustter idx of each cell
        df_stack_normalized.loc['dbscan_labels'] = df_tsne_dbscan.iloc[2]
        df_stack_log.loc['dbscan_labels'] = df_tsne_dbscan.iloc[2]
        # the mean() next line is taken on each clustter seperately
        df_stack_normalized_avg = (df_stack_normalized.T.groupby(by = 'dbscan_labels').mean()).iloc[1:] # iloc[1:] to drop the '-1' clustter idx 
        # for all cells in each clustter - sum the gene expression value for every gene 
        df_stack_frac = (df_stack_log.T.groupby(by = 'dbscan_labels').sum()).iloc[1:] # iloc[1:] to drop the '-1' clustter idx 
        # now, for each row (meaning, for each clustter) divide by the sum of gene expression of all genes
        # each gene now (meaning every value in the metrics) is a fraction of how much he is expressed 
        df_stack_frac = df_stack_frac.div(df_stack_frac.sum(axis=1), axis=0) 
        # make sure all values in sanity_checker are one - DONE
        sanity_checker = df_stack_frac.sum(axis=1)
        utils.write_log(f"saving a table of each clustter's its average cell and average expression percentage of each gene")
        df_stack_normalized_avg.T.to_csv(path_out_avg_clust_cell, sep=',')
        df_stack_frac.T.to_csv(path_out_frac, sep=',')
        utils.write_log(f"finished avg_and_fraction_clustter_expression, results saved to: {path_out_avg_clust_cell} and {path_out_frac}")


    def pca_avg_clust_cells(path_in, path_out, plots_folder='./plots_folder1', pca_dim = 10):
        df = pd.read_csv(path_in, index_col=0, header=0)
        utils.write_log(f"starting pca_avg_clust_cells: original data shape is {df.shape} (we Transpose this in a moment)")

        df_t = df.T  # TODO this way PCA is 14. according to the last call with Amit this is the right way
        pca = decomposition.PCA(n_components=pca_dim)
        principal_components = pca.fit_transform(df_t)
        explain = pca.explained_variance_

        new_cols = [f'pca_feature_{i}' for i in range(1, pca_dim + 1)]
        principal_df = pd.DataFrame(data=principal_components, columns=new_cols, index=df_t.index)
        print(f'Now PCA with {pca_dim}. explain now is:\n', explain)
        principal_df.T.to_csv(path_out, sep=',')
        utils.write_log(f'finish PCA on the avg cells of each clustter. left with {pca_dim} values. current data shape is '
                        f'{principal_df.shape} (Transposed). saved to {path_out}')

    avg_and_fraction_clustter_expression(path_in_stack='./merged_data5/stacked_3.csv',
                              path_tsne_dbscan_data='./clusttered_data/clust_tsne_data.csv', path_out_avg_clust_cell = './clusttered_data/avg_clust_cells.csv',
                              path_out_frac='./clusttered_data/frac_clust_cells.csv')

    pca_avg_clust_cells(path_in = './clusttered_data/avg_clust_cells.csv', path_out='./clusttered_data/PCA_avg_clust.csv', pca_dim=10)
    

def linkage_on_clustters(path_in, path_out, path_out_translation, plots_folder='./plots_folder1'):
    utils.write_log(f"starting linkage_on_clustters")
    df = pd.read_csv(path_in, index_col=0, header=0)
    Z = linkage(df.T, 'single', 'euclidean', True)
    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(Z)
    df_linkage_data = pd.DataFrame(Z, columns=['cluster1', 'cluster2', 'dist_btw_clust1_clust2', 'orig_observations'])
    df_clustter_idx_translation = pd.DataFrame(dn['ivl'], columns=['linkage_clustter_idx'])
    df_clustter_idx_translation.index.name = 'dbscan_clustter_idx'
    df_clustter_idx_translation.T.to_csv(path_out_translation, sep=',')
    df_linkage_data.T.to_csv(path_out, sep=',')
    data_plot_utils.save_plots(plt, f'{plots_folder}/linkage_of_clustters')
    # plt.show()
    utils.write_log(f'finished performing linkage_on_clustters, linkage output saved to: {path_out} and the indeces given to clustters by the linkage saved to {path_out_translation}')

def heatmap_data_perp():    
    def find_marker_genes(path_in_frac, path_in_avg):
        # find for each clustter it's marker genes 
        # clustters' indeces are according to the dbscan, we need to later transform them to linkage indeces!!!
        utils.write_log(f"starting find_marker_genes: currently each clustter will have at most 8 unique markers")
        df_frac = pd.read_csv(path_in_frac, index_col=0, header=0)
        df_avg = pd.read_csv(path_in_avg, index_col=0, header=0)
        
        df_frac_mean  = df_frac.mean(axis=1)
        df_avg_mean  = df_avg.mean(axis=1)
        # the best avg and frac are those who are the farthest from the average avg or frac (yes yes, average of averages for each clustter)
        df_frac_normalized = ((df_frac).sub(df_frac_mean, axis=0))
        df_avg_normalized = ((df_avg).sub(df_avg_mean, axis=0))
        # in both normalized and loaded csvs' , the clustters are columns and the genes are rows 
        marker_genes_dict = {}
        selected_marker_genes = []
        for clustter_idx in df_avg.columns:
            # find the 4 largest avg and frac w.r.t. all others avg and frac of other clustters
            marker_genes_frac = df_frac_normalized[clustter_idx][df_frac_normalized[clustter_idx] > 0].nlargest(4)
            marker_genes_avg = df_avg_normalized[clustter_idx][df_avg_normalized[clustter_idx] > 0].nlargest(4)
            marker_idx_genes_avg = marker_genes_avg.index.to_list()
            marker_idx_genes_frac = marker_genes_frac.index.to_list()
            # set cancels repetitions of genes in the avg and the frac lists
            unique_curr_marker_genes = set(marker_idx_genes_avg + marker_idx_genes_frac)
            # make sure we haven't selected these genes already in the past
            # how to subtract one list from another: set(unique_curr_marker_genes) - set(selected_marker_genes)
            unique_curr_marker_genes = set(unique_curr_marker_genes) - set(selected_marker_genes)
            clustter_idx = int(float(clustter_idx))
            marker_genes_dict[clustter_idx] = unique_curr_marker_genes
            # make sure to keep all genes we already chose
            selected_marker_genes = selected_marker_genes + list(unique_curr_marker_genes)
        utils.write_log(f"finished find_marker_genes: returning a dictionary [keys - dbscan clustter idx, values - idx of marker genes]")
        return marker_genes_dict

    def translate_and_sort_dict_keys(marker_dict_dbscan_idx, path_in_translation='./clusttered_data/clust_idx_translation_table.csv'):
        utils.write_log(f"starting translate_and_sort_dict_keys")
        df_translation = pd.read_csv(path_in_translation, index_col=0, header=0)
        marker_dict_linkage_idx = {}
        sorted_dict = {}
        for i in df_translation.columns:
            marker_dict_linkage_idx[df_translation[i][0]] = marker_dict_dbscan_idx.pop(int(i))
        for key in sorted(marker_dict_linkage_idx):
            sorted_dict[key] = marker_dict_linkage_idx[key]
        utils.write_log(f"finished translate_and_sort_dict_keys: returning a sorted(!) dictionary [keys - linkage(!) clustter idx, values - idx of marker genes]")
        return sorted_dict

    def translate_clustter_data(path_in_clustter_data, path_in_translation, path_out_clustter_data ='./clusttered_data/clust_idx_translation_table.csv'):
        utils.write_log(f"starting translate_clustter_data")
        df_translation = pd.read_csv(path_in_translation, index_col=0, header=0)
        df_clust_data = pd.read_csv(path_in_clustter_data, index_col=0, header=0)
        # initialization of the new row
        # we'll soon replace the dbscan_labels values in this row
        df_clust_data.loc['linkage_labels'] = df_clust_data.loc['dbscan_labels']
        for i in df_clust_data.columns:
            col_in_translation_table = str(int(df_clust_data[i]['dbscan_labels']))
            if col_in_translation_table == '-1':
                df_clust_data[i]['linkage_labels'] = -1
            else:
                df_clust_data[i]['linkage_labels'] = df_translation[col_in_translation_table][0]
        df_clust_data.to_csv(path_out_clustter_data, sep=',')
        utils.write_log(f"finished translate_clustter_data: translation table of clustter indeces saved to {path_out_clustter_data}")


    def sort_stack_data_by_clustter(path_in_clustter_data = './clusttered_data/clust_tsne_data.csv', path_in_stack_data='./merged_data5/stacked_3.csv',
            path_out = './clusttered_data/stacked_3_sort_by_clust.csv'):
        utils.write_log(f"starting sort_stack_data_by_clustter")
        df_clust_data = pd.read_csv(path_in_clustter_data, index_col=0, header=0)
        df_stack_data = pd.read_csv(path_in_stack_data, index_col=0, header=0)
        df_stack_data_T = df_stack_data.T
        df_stack_data_T['linkage_labels'] = df_clust_data.loc['linkage_labels']
        df_stack_data_T.sort_values(by=['linkage_labels'], inplace=True)
        df_stack_data_T.T.to_csv(path_out, sep=',')
        utils.write_log(f"finished sort_stack_data_by_clustter: mapping each cell to its clustter linkage idx instead dbscan clustter idx ")
        utils.write_log(f"result of sort_stack_data_by_clustter saved to {path_out}")
        pass

    def prepare_data_for_heatmap(sorted_marker_genes_dict, path_in_stack_data, path_out='./clusttered_data/stacked_3_for_heatMap.csv'):
        # combine all the marker genes to one long list
        # this is the order they should appear in the heat map rows
        # they are already ordered as markers for the clustters' linkage idxs in ascending order (happened in translate_and_sort_dict_keys)
        utils.write_log(f"starting prepare_data_for_heatmap")
        marker_genes_by_order = []
        for curr_markers in sorted_marker_genes_dict.values():
            marker_genes_by_order = marker_genes_by_order + list(curr_markers)

        df_stack_data = pd.read_csv(path_in_stack_data, index_col=0, header=0)
        # some of the indeces are integers and some are strings (problem!) so we make them all strings and the marker genes list also
        marker_genes_by_order = map(str, marker_genes_by_order)
        df_stack_data.index = df_stack_data.index.map(str)
        # keep only rows (meaning, genes) who are marker genes of the clustters
        df_stack_data = df_stack_data.loc[marker_genes_by_order]
        # order the rows (meaning, genes) in the order they appear in "marker_genes_by_order"
        df_stack_data.reindex(marker_genes_by_order)
        # df_stack_data.to_csv('./clusttered_data/stacked_3_for_heatMap_check.csv', sep=',')
        df_stack_data_log = np.log2(df_stack_data+1)
        df_stack_data_normalized = df_stack_data_log.sub(df_stack_data_log.mean(axis=1), axis=0)
        df_stack_data_normalized = df_stack_data_normalized.div(df_stack_data_log.std(axis=1), axis=0)
        df_stack_data_normalized.to_csv(path_out, sep=',')
        utils.write_log(f"finished prepare_data_for_heatmap: marker genes are order accoring to linkage clustters, gene expressions are normalized ")
        utils.write_log(f"result of prepare_data_for_heatmap saved to {path_out}")

    # here, the dict keys are still according to the dbscan clustter indeces
    marker_genes_dict = find_marker_genes(path_in_frac = './clusttered_data/frac_clust_cells.csv', path_in_avg= './clusttered_data/avg_clust_cells.csv')
    
    # now, the keys will be translated to linkage clustter idx and sorted accordingly from 0 to [num_of clustters-1]  
    marker_dict_linkage_idx = translate_and_sort_dict_keys(marker_dict_dbscan_idx= marker_genes_dict, path_in_translation='./clusttered_data/clust_idx_translation_table.csv')
    
    # same goes for the stacked data, need to translate to linkage clusster idx and sort accordingly
    translate_clustter_data(path_in_clustter_data = './clusttered_data/clust_tsne_data.csv', path_in_translation='./clusttered_data/clust_idx_translation_table.csv',
                            path_out_clustter_data='./clusttered_data/clust_tsne_data.csv')
    
    sort_stack_data_by_clustter(path_in_clustter_data = './clusttered_data/clust_tsne_data.csv', path_in_stack_data='./merged_data5/stacked_3.csv',
                                path_out = './clusttered_data/stacked_3_sort_by_clust.csv')

    # now, final preperations to create the .csv table for the heatmap
    prepare_data_for_heatmap(path_in_stack_data = './clusttered_data/stacked_3_sort_by_clust.csv', sorted_marker_genes_dict = marker_dict_linkage_idx,
                    path_out = './clusttered_data/stacked_3_for_heatMap.csv')

def create_heatmap(path_in_heatmap_table='./clusttered_data/stacked_3_for_heatMap.csv', plots_folder = './plots_folder1/testing2_out'):
    # plt.figure(figsize=(10, 10))
    utils.write_log(f"starting create_heatmap")
    df_stack_data_heatmap = pd.read_csv(path_in_heatmap_table, index_col=0, header=0)
    vmin_val = df_stack_data_heatmap.to_numpy().min()
    vmax_val = df_stack_data_heatmap.to_numpy().max()
    # sns.heatmap(df_stack_data_normalized, annot=True, cmap = cm.RdBu, vmin=vmin_val, vmax=vmax_val)
    # plt.title("Heatmap for gene expression of Marker Genes of all clusters ordered by Linkage")
    # plt.show()
    fig, ax = plt.subplots(figsize=(24, 18))

    hm = sns.heatmap(df_stack_data_heatmap, cbar=True, vmin=vmin_val, vmax=vmax_val,
                    fmt='.2f', annot_kws={'size': 3}, annot=True, 
                    square=True, cmap=plt.cm.RdBu)

    ticks = np.arange(df_stack_data_heatmap.shape[0]) + 0.5
    ax.set_xticks(ticks)
    ax.set_xticklabels(df_stack_data_heatmap.columns, rotation=90, fontsize=8)
    ax.set_yticks(ticks)
    ax.set_yticklabels(df_stack_data_heatmap.index, rotation=360, fontsize=8)

    ax.set_title('Heatmap for gene expression of Marker Genes of all clusters ordered by Linkage')
    plt.tight_layout()
    plt.show()
    # need to change to :
    # data_plot_utils.save_plots(plt, f'{plots_folder}/heatmap')
    # for experiments sake, tried only to save it in a .png format
    plt.savefig(str(plots_folder)+"/corr_matrix_incl_anno_double.png", dpi=300)
    utils.write_log(f"finished create_heatmap: result of create_heatmap saved to {plots_folder}")
    pass

def linkage_pipeline():
    # look for further explanations and comments in the wrapper functions 
    utils.write_log(f"#### starting linkage_pipeline ####")
    sanity_checks(path_in_stack='./merged_data5/stacked_3.csv',
                    path_in_dbscan='./clusttered_data/dbscan.csv',  
                    gene_list= ['Snap25','Gad2','Slc32a1', 'Slc17a7','Slc17a6','Sst','Tac2','Acta2','Flt1','Cldn5', 'Aqp4','Plp1'],
                    path_to_features_csv='./csv_data2/features.csv', plots_folder = './plots_folder1/testing2_out')

    linkage_data_prep()

    linkage_on_clustters(path_in='./clusttered_data/PCA_avg_clust.csv', path_out='./clusttered_data/linkage_out.csv', path_out_translation='./clusttered_data/clust_idx_translation_table.csv')
    utils.write_log(f"#### finished linkage_pipeline ####")

def heatmap_pipeline():
    utils.write_log(f"#### starting heatmap_pipeline ####")
    heatmap_data_perp()

    # finally, compute the heatmap
    create_heatmap(path_in_heatmap_table='./clusttered_data/stacked_3_for_heatMap.csv', plots_folder = './plots_folder1/testing2_out')
    utils.write_log(f"#### finished heatmap_pipeline ####")

# if __name__=="__main__":
#     create_heatmap(path_in_heatmap_table='./clusttered_data/stacked_3_for_heatMap.csv', plots_folder = './plots_folder1/testing2_out')