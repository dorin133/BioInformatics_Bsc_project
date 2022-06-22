import pandas as pd
import numpy as np
import seaborn as sns
import data_plot_utils
import utils
from matplotlib import pyplot as plt
from distinctipy import distinctipy


def clustter_nueronal_genes(path_to_features, path_frac_clust_cells, path_in_cluseters, path_out):
    utils.write_log(f"starting clustter_nueronal_genes: nueral gene expression on T-SNE's scatter plot")

    features = pd.read_csv(path_to_features, header=0)
    df_frac_clust_cells = pd.read_csv(path_frac_clust_cells, header=0, index_col=0)
    features.set_axis(['num', 'geneID', 'geneName'], axis=1, inplace=True)
    features.set_index('geneName', inplace=True)
    exclude_markers_names = {'C1qc', 'C1qa', 'C1qb', 'Gja1', 'Cx3cr1', 'Acta2', 'Ly6c1', 'Mfge8', 'Plxnb3', 'Cldn11', 'Aqp4',
                       'Vtn', 'Cldn5', 'Pdgfrb', 'Flt1', 'Slc25a18', 'Pdgfra', 'Foxj1', 'Olig1', 'Olig2', 'Sox10',
                       'Hbb-bs', 'Hbb-bt', 'Hba-a2', 'Ttr'}  # TODO
    exclude_markers_ids = []
    for gene_name in exclude_markers_names:
        gene_id = int(features.at[gene_name, 'num'])
        if gene_id not in df_frac_clust_cells.index:
            print(f"!!! exclude marker {gene_name} (gen id {gene_id}) is not in the stacked matrix data")
            continue
        print(f'exclude marker gen id for {gene_name} is {gene_id}')
        exclude_markers_ids.append(gene_id)
    print(df_frac_clust_cells.shape)
    sum_exclude_markers = df_frac_clust_cells.T[exclude_markers_ids].sum(1).T
    print(df_frac_clust_cells.index[-1])
    df_frac_clust_cells.loc[len(df_frac_clust_cells.index)+1] = sum_exclude_markers
    print(df_frac_clust_cells.index[-1])

    nueronal_genes_name = {
        'Gad2': 'Gaba',
        'Slc17a7': 'Glut1',
        'Slc17a6': 'Glut2'
    }

    glut1_id = 999
    glut2_id = 999
    nueronal_genes_idx = {}
    found_genes = []
    not_in_data_gens = []
    counter = 0
    for gene_name in nueronal_genes_name:
        gene_id = int(features.at[gene_name, 'num'])
        if gene_id not in df_frac_clust_cells.index:
            print(f"!!! gen {gene_name} (gen id {gene_id}) is not in the stacked matrix data")
            not_in_data_gens.append((gene_name, gene_id))
            continue
        print(f'gen id for {gene_name} is {gene_id}')
        if gene_name == 'Slc17a7':
            glut1_id = counter
        if gene_name == 'Slc17a6':
            glut2_id = counter
        found_genes.append(gene_id)
        nueronal_genes_idx[counter] = nueronal_genes_name[gene_name]
        counter += 1
    found_genes.append(df_frac_clust_cells.index[-1])
    nueronal_genes_idx[counter] = 'Non Nueronal'
    counter += 1
    doublets_idx = counter
    nueronal_genes_idx[counter] = 'Doublets'

    # print(found_genes)
    df_nueronal_frac = (df_frac_clust_cells.loc[found_genes])
    # print(df_nueronal_frac.shape)
    # print(df_nueronal_frac)
    clust_nueral_class = {}
    df_nueronal_frac.index.name = 'gene_id'
    for col in df_nueronal_frac.columns:
        nueral_frac_arr = df_nueronal_frac[col].to_numpy()
        # print(nueral_frac_arr)
        two_largest = df_nueronal_frac[col].nlargest(2)
        # print('two_largest:\n', two_largest)
        two_largest_idx = (df_nueronal_frac[col].nlargest(2).index).tolist()
        if [found_genes[glut1_id], found_genes[glut2_id]] == two_largest_idx or \
                [found_genes[glut2_id], found_genes[glut1_id]] == two_largest_idx:
            clust_nueral_class[col] = nueral_frac_arr.argmax()
        elif two_largest.max() < two_largest.min() * 2:  # search for doublets
        # elif nueral_frac_arr.max() < two_largest_idx.min() * 2:  # search for doublets
            clust_nueral_class[col] = doublets_idx
        else:
            clust_nueral_class[col] = nueral_frac_arr.argmax()

    map_nueral_class = {float(k): nueronal_genes_idx[v] for k, v in clust_nueral_class.items()}
    map_nueral_class[-1.0] = 'noise cluster'
    clusters_df = pd.read_csv(path_in_cluseters, header=0, index_col=0).T
    clusters_df['nueral_labels'] = clusters_df['linkage_labels'].map(map_nueral_class)
    clusters_df = clusters_df.T
    clusters_df.to_csv(path_out, sep=',')
    # clusters_df.value_counts()
    utils.write_log(f"finished clustter_nueronal_genes {clusters_df.T['nueral_labels'].value_counts().to_dict()}")

    plot_nueral_gene_expression(path_clust_tsne_data=path_out)


def plot_nueral_gene_expression(path_clust_tsne_data, plots_folder='plots_folder1/part2'):
    utils.write_log(f"starting plot_nueral_gene_expression")
    df_tsne = pd.read_csv(path_clust_tsne_data, index_col=0, header=0).T
    df_tsne = df_tsne[df_tsne['nueral_labels'] != 'noise cluster']
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
#
#
# def translate_clustter_data(path_in_clustter_data, nueral_idx_arr, path_out_clustter_data ='./clusttered_data/clust_idx_translation_table.csv'):
#         utils.write_log(f"starting translate_clustter_data : adding nueral gene clustter index")
#         # df_translation = pd.read_csv(path_in_translation, index_col=0, header=0)
#         df_clust_data = pd.read_csv(path_in_clustter_data, index_col=0, header=0)
#         # initialization of the new row
#         # we'll soon replace the dbscan_labels values in this row
#         df_clust_data.loc['nueral_labels'] = 4
#         for i in df_clust_data.columns:
#             col_in_translation_table = int(df_clust_data[i]['linkage_labels'])
#             if not col_in_translation_table == -1:
#                 df_clust_data[i]['nueral_labels'] = nueral_idx_arr[col_in_translation_table]
#         df_clust_data.to_csv(path_out_clustter_data, sep=',')
#         utils.write_log(f"finished translate_clustter_data: translation table of nueral gene clustter indeces saved to {path_out_clustter_data}")


if __name__ == '__main__':
    # avg_and_fraction_clustter_expression(path_in_stack='./merged_data5/stacked_1.csv',
    #                                      path_tsne_dbscan_data='./clusttered_data/clust_tsne_data.csv',
    #                                      path_out_avg_clust_cell = './clusttered_data/avg_clust_cells_stk1.csv',
    #                                      path_out_frac='./clusttered_data/frac_clust_cells_stk1.csv')
    # arr = clustter_nueronal_genes(path_to_features='./csv_data2/features.csv',
    #                               path_frac_clust_cells='./clusttered_data/frac_clust_cells_stk1.csv',
    #                               path_in_cluseters='./clusttered_data/clust_tsne_data.csv',
    #                               path_out='./clusttered_data/tsne_and_clust_labels.csv')
    # plot_nueral_gene_expression(path_clust_tsne_data='./clusttered_data/tsne_and_clust_labels.csv',
    #                             plots_folder='./plots_folder1/testing2_out')
    clustter_nueronal_genes(path_to_features='./csv_data2/features.csv',
                                  path_frac_clust_cells='./clusttered_data6/frac_clust_cells_stk1.csv',
                                  path_in_cluseters='./clusttered_data6/clust_tsne_data.csv',
                                  path_out='./clusttered_data6/tsne_and_clust_labels.csv')
    print("Done")
