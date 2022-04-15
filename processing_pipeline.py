import utils
import data_processing
import data_plot_utils


def main():

    #step 1: turn each features.tsv file (list of the genes) to a pandas .csv file
    data_processing.features_to_csv(folder_path='./raw_data', out_folder_path='./raw_csv_data')

    #step 2: turn each .mtx file (list of the future matrix values) to a matrix(!) .csv file
    utils.run_func_for_all(
        func_to_run=data_processing.raw_mtx_to_csv,
        folder_in_path='./raw_data',
        folder_out_path='./raw_csv_data',
        which_files=lambda x: '.mtx' in x,
        rename_out_files=lambda x: x[:-4] + '.csv'
    )

    #step 3: create the metadata files for each matrix .csv file
    data_processing.prepare_metadata_single_files(folder_path='./raw_data', out_folder_path='./raw_csv_data')  # TODO

    """ up until here, we created and put the matrix files, feature file and metadata files from the 'raw_data' folder to the 'raw_csv_data' folder"""
    #step 4: filter both matrices and metadatas of each sample seperately
    utils.run_func_for_all(
        func_to_run=data_processing.filter_cols,
        folder_in_path='./raw_csv_data',
        folder_out_path='./filtered_data',
        which_files=lambda x: '_matrix.csv' in x,
        rename_out_files=lambda x: x[:-4] + '_filtered.csv'
    )

    data_processing.filter_metadata_rows(folder_mtx_path='./filtered_data', folder_to_metadata='./raw_csv_data',
                                         out_folder_path='./filtered_data')  # TODO

    """ now, our most relevant matrix files and metadata files are in the 'filtered_data.csv' folder"""
    #plots for step 4
    data_plot_utils.print_hist_mul("./filtered_data")
    data_plot_utils.print_hist_mt_percentage(features_folder_path='./raw_csv_data', folder_path='./filtered_data')

    # step 5: normalize the data to make the samples' euclidian distance 20,000
    utils.run_func_for_all(
        func_to_run=data_processing.normalize_data,
        folder_in_path='./filtered_data',
        folder_out_path='./normalized_mtx',
        which_files=lambda x: 'matrix_filtered.csv' in x,
    )
    
    """ after mtx normalization, our most relevant matrix files files are in the 'filtered_data.csv' folder"""
    """ as to other relevant files: the relevant metadatas are in 'filtered_data' 
        and features.csv is in raw_csv_data
        
        hope this order makes sense...if not, feel free to correct me"""

    # step 6: after filtering and normalizing separately, merge all files together
    utils.stack_csv_together(folder_path='./normalized_mtx', out_file_path='./merged_data/stacked_normalized_mtx.csv')
    utils.merge_all_metadata(folder_path='./filtered_data', out_folder_path='./merged_data')
    
    """ final stacked versions of both the metdata and the matrices are in the 'merged_data' file"""

    # #step 7: plot and calculate the mean w.r.t. the coe. of variance for each gene
    data_processing.calc_and_plot_cv(path_stacked_mtx_file='./merged_data/stacked_normalized_mtx.csv', path_to_features_csv='./raw_csv_data/features.csv')


    pass


if __name__ == '__main__':
    main()
    print("Done")

