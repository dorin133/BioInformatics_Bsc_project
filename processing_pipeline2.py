import utils
import data_processing
import data_plot_utils
import ml_processing
import pandas as pd
import linkage_and_heatmap as link_and_heat
import gaba_genes_processing
import data_processing2


def main():

    # utils.write_log('*********************************** Start pipeline run (v2) ***********************************')
    #
    # data_processing2.filter_gaba_only(path_clust_labels='./clusttered_data6/tsne_and_clust_labels.csv',
    #                                   folder_path_in='./csv_data2', folder_path_out='./csv_gaba7')
    #
    # # TODO filtering
    #
    # # TODO feature selection
    #
    # utils.stack_csv_together(folder_path='./csv_gaba7', out_file_path='./cluster_gaba8/stack_gaba.csv',
    #                          filters=lambda x: 'gaba_matrix.csv' in x)
    #
    # data_processing2.filter_rare_gens(path_stacked_mtx_file='./cluster_gaba8/stack_gaba.csv',
    #                                   path_out_file='./cluster_gaba8/stack_filtered_gaba.csv')
    #
    # data_processing.pca_norm_knee(path_in='./cluster_gaba8/stack_filtered_gaba.csv',
    #                               path_out='./cluster_gaba8/pca_gaba.csv', plots_folder='./plots_folder1/part3')
    #
    # ml_processing.tSNE(path_in='./cluster_gaba8/pca_gaba.csv', path_to_MEA='./raw_data/MEA_dimorphism_samples.xlsx',
    #                    path_out='./cluster_gaba8/tsne_gaba.csv', plots_folder='./plots_folder1/part3')
    #
    # ml_processing.DBScan_dynm_eps(eps_prc=70, k_neighbor=20, path_in='./cluster_gaba8/tsne_gaba.csv',
    #                               path_out='./cluster_gaba8/dbscan_gaba.csv',
    #                               path_out_tsne_dbscan='./cluster_gaba8/clust_tsne_gaba.csv',
    #                               print_noise=False, plots_folder='./plots_folder1/part3')

    data_processing2.sanity_checks_gaba(path_in='./cluster_gaba8/dbscan_gaba.csv',
                                        path_to_MEA = './raw_data/MEA_dimorphism_samples.xlsx',
                                        print_noise=False, plots_folder='./plots_folder1/part3')

    utils.write_log('*********************************** Finish pipeline run (v2) ***********************************')


if __name__ == '__main__':
    main()
    print("Done")
