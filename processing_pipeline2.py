import utils
import data_processing
import data_plot_utils
import ml_processing
import pandas as pd
import linkage_and_heatmap as link_and_heat
import os
import gaba_genes_processing
import data_processing2


def main():

    utils.write_log('*********************************** Start pipeline run (v2) ***********************************')
    
    # utils.check_files_and_folder_for_complete_run(first_folder="./raw_data")
    
    # data_processing2.filter_gaba_only(path_clust_labels='./clusttered_data6/tsne_and_clust_labels.csv',
    #                                   folder_path_in='./csv_data2', folder_path_out='./csv_gaba7')
    
    
    # # data_processing.features_to_csv(folder_path='./raw_data', out_folder_path='./csv_data2')  # no need to run again
    # # # step 2: turn each .mtx file (list of the future matrix values) to a matrix(!) .csv file  # no need to run again
    # # utils.run_func_for_all(
    # #     func_to_run=data_processing.raw_mtx_to_csv,
    # #     folder_in_path='./raw_data',
    # #     folder_out_path='./csv_data2',
    # #     which_files=lambda x: '.mtx' in x,
    # #     rename_out_files=lambda x: x[:-4] + '.csv'
    # # )
    # # data_processing.prepare_metadata_single_files(folder_path='./raw_data', out_folder_path='./csv_data2')    # no need to run again
    #
    # """ up until here, we created and put the matrix files, feature file and metadata files from the 'raw_data' folder to the 'raw_csv_data' folder"""
    # # step 4: filter both matrices and metadatas of each sample seperately
    # utils.run_func_for_all(
    #     func_to_run=data_processing.filter_cols,
    #     folder_in_path='./csv_gaba7',
    #     folder_out_path='./gaba_filtered_data8',
    #     which_files=lambda x: '_matrix.csv' in x,
    #     rename_out_files=lambda x: x[:-4] + '_filtered.csv'
    # )
    #
    # data_processing.filter_metadata_rows(folder_mtx_path='./gaba_filtered_data8', folder_to_metadata='./csv_data2',
    #                                      out_folder_path='./gaba_filtered_data8')
    #
    # """ now, our most relevant matrix files and metadata files are in the 'filtered_data.csv' folder"""
    # # plots for step 4
    # data_plot_utils.print_hist_genes(folder_path='./csv_gaba7', plots_folder='./plots_folder1/part3')
    # data_plot_utils.print_pdf_mul("./gaba_filtered_data8", plots_folder='./plots_folder1/part3')
    # data_plot_utils.print_hist_mt_percentage(features_folder_path='./csv_data2', folder_path='./gaba_filtered_data8',
    #                                          plots_folder='./plots_folder1/part3')
    
    # step 5: normalize the data to make the samples' euclidian distance 20,000
    # utils.run_func_for_all(
    #     func_to_run=data_processing.normalize_data,
    #     folder_in_path='./gaba_filtered_data8',
    #     folder_out_path='./gaba_normalized_data9',
    #     which_files=lambda x: 'matrix_filtered.csv' in x,
    # )
    #
    # """ after mtx normalization, our most relevant matrix files are in the 'filtered_data.csv' folder"""
    # """ as to other relevant files: the relevant metadatas are in 'filtered_data'
    #     and features.csv is in raw_csv_data
    #
    #     hope this order makes sense...if not, feel free to correct me"""
    # step 6: after filtering and normalizing separately, merge all files together
    # utils.stack_csv_together(folder_path='./gaba_normalized_data9', out_file_path='./gaba_merged_data10/gaba_stacked_1.csv')
    # utils.merge_all_metadata(folder_path='./gaba_filtered_data8', out_file='./gaba_merged_data10/gaba_all_samples_metadata.csv')
    #
    # """ final stacked versions of both the metadata and the matrices are in the 'merged_data' file"""
    # # step 7: creating separate csv files for males and females
    # utils.split_merged_into_M_F(path_stacked_file='./gaba_merged_data10/gaba_stacked_1.csv',
    #                             mea_samples='./raw_data/MEA_dimorphism_samples.xlsx',
    #                             out_file_M='./gaba_merged_data10/gaba_stacked_M.csv',
    #                             out_file_F='./gaba_merged_data10/gaba_stacked_F.csv')
    
    # step 8: plotting and pointing the most diff genes for males vs. females
    # data_plot_utils.plot_female_vs_male_mean(females_path='./gaba_merged_data10/gaba_stacked_F.csv',
    #                                          males_path='./gaba_merged_data10/gaba_stacked_M.csv',
    #                                          path_to_features_csv='./csv_data2/features.csv',
    #                                          path_stacked_mtx_file='?????Not yet imp?????',  # TODO
    #                                          path_out='????Not yet imp?????',
    #                                          plots_folder='./plots_folder1/part3')
    
    # data_plot_utils.plot_female_vs_male_fraction_expression(females_path='./gaba_merged_data10/gaba_stacked_F.csv',
    #                                                         males_path='./gaba_merged_data10/gaba_stacked_M.csv',
    #                                                         path_to_features_csv='./csv_data2/features.csv',
    #                                                         path_stacked_mtx_file='?????Not yet imp?????',  # TODO
    #                                                         path_out='????Not yet imp?????',
    #                                                         plots_folder='./plots_folder1/part3')
    
    # # step 9: filter the whole stacked mtx's very common genes and very rare ones (now for the first time in the code, we filter rows and not columns + no need to adjust metadata)
    # data_processing.filter_common_and_rare_gens(path_stacked_mtx_file='./gaba_merged_data10/gaba_stacked_1.csv',
    #                                             path_out_file='./gaba_merged_data10/gaba_stacked_2.csv')  # TODO VVV which one to take??
    # data_processing.filter_common_and_rare_gens(path_stacked_mtx_file='./gaba_merged_data10/gaba_stacked_1.csv', # TODO ^^
    #                                             path_out_file='./gaba_merged_data10/gaba_stacked_2_v2.csv',
    #                                             do_no_filter_gens=[1868])  # 1868 is gaba (gad2)
    #
    #
    # step 10: plot and calculate the mean w.r.t. the coe. of variance for each gene
    # data_processing.calc_and_plot_cv(path_stacked_mtx_file='./gaba_merged_data10/gaba_stacked_2.csv',
    #                                  path_out='./gaba_merged_data10/gaba_stacked_3.csv',
    #                                  path_to_features_csv='./csv_data2/features.csv',
    #                                  plots_folder='./plots_folder1/part3')
    
    # # step 11: PCA
    # data_processing.pca_norm_knee(path_in='./gaba_merged_data10/gaba_stacked_3.csv',
    #                               path_out='./gaba_merged_data10/gaba_pca4.csv', plots_folder='./plots_folder1/part3')
    #
    # step 12: tSNE
    # ml_processing.tSNE(path_in='./gaba_merged_data10/gaba_pca4.csv',
    #                    path_to_MEA='./raw_data/MEA_dimorphism_samples.xlsx',
    #                    path_out='./gaba_merged_data10/gaba_tsne.csv', plots_folder='./plots_folder1/part3')
    
    # # step 13: DBScan
    # ml_processing.DBScan_dynm_eps(eps_prc=70, k_neighbor=20, path_in='./gaba_merged_data10/gaba_tsne.csv',
    #                               path_out='./gaba_clustered_data11/gaba_dbscan.csv',
    #                               path_out_tsne_dbscan='./gaba_clustered_data11/gaba_clust_tsne_data.csv',
    #                               plots_folder='./plots_folder1/part3')
    #
    # # step 14: sanity_checks
    # link_and_heat.sanity_checks(path_in_stack='./gaba_merged_data10/gaba_stacked_1.csv',
    #                             path_in_dbscan='./gaba_clustered_data11/gaba_dbscan.csv',
    #                             path_to_features_csv='./csv_data2/features.csv',
    #                             gene_list= ['Gad2', 'Slc17a6', 'Slc17a7'] ,#['Snap25', 'Gad2', 'Slc32a1', 'Slc17a7', 'Slc17a6', 'Sst', 'Tac2', 'Acta2',
    #                                        #'Flt1', 'Cldn5', 'Aqp4', 'Plp1'],
    #                             plots_folder='./plots_folder1/part3')
    
    # data_processing2.sanity_checks_gaba(path_in='./gaba_clustered_data11/gaba_dbscan.csv',
    #                                     path_to_MEA='./raw_data/MEA_dimorphism_samples.xlsx',
    #                                     print_noise=False, plots_folder='./plots_folder1/part3')
    #
    #
    # # step 15: linkage data prep and the linkage step itself
    # link_and_heat.linkage_pipeline()
    # link_and_heat.linkage_pipeline(path_in_stack='./gaba_merged_data10/gaba_stacked_2.csv',
    #                                path_in_dbscan='./gaba_clustered_data11/gaba_clust_tsne_data.csv',
    #                                path_out_avg_clust_cell='./gaba_clustered_data11/gaba_avg_clust_cells.csv',
    #                                path_out_frac='./gaba_clustered_data11/gaba_frac_clust_cells.csv',
    #                                path_out_pca_avg='./gaba_clustered_data11/gaba_PCA_avg_clust.csv',
    #                                path_out_linkage='./gaba_clustered_data11/gaba_linkage_out.csv',
    #                                path_out_translation='./gaba_clustered_data11/gaba_clust_idx_translation_table.csv',
                                #    plots_folder='./plots_folder1/part3')
    
    # # step 16: heatmap data prep and the heatmap step itself
    # link_and_heat.heatmap_pipeline()
    # link_and_heat.heatmap_pipeline(path_in_frac='./gaba_clustered_data11/gaba_frac_clust_cells.csv',
    #                                path_in_avg='./gaba_clustered_data11/gaba_avg_clust_cells.csv',
    #                                features_csv_path='./csv_data2/features.csv',
    #                                path_in_translation='./gaba_clustered_data11/gaba_clust_idx_translation_table.csv',
    #                                path_in_clustter_data='./gaba_clustered_data11/gaba_clust_tsne_data.csv',
    #                                path_in_stack_data='./gaba_merged_data10/gaba_stacked_2.csv',
    #                                path_out_sort_stack_data='./gaba_clustered_data11/gaba_stacked_2_sort_by_clust.csv',
    #                                path_out_stack_for_heatmap='./gaba_clustered_data11/gaba_stacked_2_for_heatMap.csv',
    #                                plots_folder='./plots_folder1/part3')
    
    # # TODO that probably should be deleted VVV
    # # TODO 2: something in the output just looks strange... i thought all gonna be gaba but they are not
    # step 17: gaba_genes_processing
    # gaba_genes_processing.avg_and_fraction_clustter_expression(path_in_stack='./gaba_merged_data10/gaba_stacked_1.csv',
    #                                                            path_tsne_dbscan_data='./gaba_clustered_data11/gaba_clust_tsne_data.csv',
    #                                                            path_out_avg_clust_cell='./gaba_clustered_data11/gaba_avg_clust_cells_stk1.csv',
    #                                                            path_out_frac='./gaba_clustered_data11/gaba_frac_clust_cells_stk1.csv')
    
    # gaba_genes_processing.clustter_nueronal_genes(path_to_features='./csv_data2/features.csv',
    #                                               path_frac_clust_cells='./gaba_clustered_data11/gaba_frac_clust_cells_stk1.csv',
    #                                               path_in_cluseters='./gaba_clustered_data11/gaba_clust_tsne_data.csv',
    #                                               path_out='./gaba_clustered_data11/gaba_tsne_and_clust_labels.csv',
    #                                               plots_folder='./plots_folder1/part3')


    # data_processing2.clusters_bar_groups(path_in='./gaba_clustered_data11/gaba_tsne_and_clust_labels.csv',
    #                                      path_to_MEA='./raw_data/MEA_dimorphism_samples.xlsx',
    #                                      plots_folder='./plots_folder1/part3')


    # step 18: gathering some informative info about the gaba  genes
    gaba_genes_processing.clustter_stats_2marker_genes(path_in_frac='./gaba_clustered_data11/gaba_frac_clust_cells_stk1.csv',
                                    path_in_avg='./gaba_clustered_data11/gaba_avg_clust_cells_stk1.csv', 
                                    max_genes_amount = 2,
                                    path_in_translation='./gaba_clustered_data11/gaba_clust_idx_translation_table.csv',
                                    path_to_features='./csv_data2/features.csv',
                                    path_tsne_dbscan_data='./gaba_clustered_data11/gaba_clust_tsne_data.csv', 
                                    path_out='./gaba_clustered_data11/gaba_all_clustter_stats.csv')
    
    def x(path_in='./gaba_clustered_data11/gaba_all_clustter_stats.csv',folder_path_in='./csv_gaba7', path_to_MEA='./raw_data/MEA_dimorphism_samples.xlsx'):
        utils.write_log(f'start filter_gaba_only')
        clust_stats = pd.read_csv(path_in, index_col=0, header=0)

        raw_files = os.listdir(folder_path_in)  # list all raw files
        chosen_files = list(filter(lambda x: 'matrix.csv' in x, raw_files))
        chosen_files.sort()

        df_f_m_index = pd.read_excel(path_to_MEA)
        f_list, m_list = [], []
        p_list, no_p_list = [], []
        for _, row in df_f_m_index.iterrows():
            if row['female'] == 1:
                f_list.append(row.iloc[0])
            else:
                m_list.append(row.iloc[0])
            if row['parent'] == 1:
                p_list.append(row.iloc[0])
            else:
                no_p_list.append(row.iloc[0])
        
        gender_per_cell = []
        paernt_or_not_per_cell = []
        for col_name in clust_stats.columns:
            tmp = col_name.split('__')[1]
            if tmp in f_list:
                gender_per_cell.append(1)
            else:
                gender_per_cell.append(0)
            if tmp in p_list:
                paernt_or_not_per_cell.append(1)
            else:
                paernt_or_not_per_cell.append(0)

        clust_stats_T = clust_stats.T
        clust_stats_T['female'] = pd.Series(gender_per_cell, index=clust_stats_T.index)
        clust_stats_T['parent'] = pd.Series(paernt_or_not_per_cell, index=clust_stats_T.index)
        path_out = path_in #cause we want to keep adding to this table
        clust_stats_T.T.to_csv(path_out, sep=',')
        utils.write_log(f"finished identifying clusters' cells by gender and parenthood for GABA stats, results saved to: {path_out}")


    x(path_in='./gaba_clustered_data11/gaba_all_clustter_stats.csv',folder_path_in='./csv_gaba7', path_to_MEA='./raw_data/MEA_dimorphism_samples.xlsx')
    utils.write_log('*********************************** Finish pipeline run (v2) ***********************************')


if __name__ == '__main__':
    main()
    print("Done")
