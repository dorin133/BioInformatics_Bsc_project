import utils
import data_processing
import data_plot_utils


def main():

    #step 1: turn each features.tsv file (list of the genes) to a pandas .csv file
    # data_processing.features_to_csv(folder_path='./raw_data_for_tests', out_folder_path='./raw_csv_data_for_tests')

    # #step 2: turn each .mtx file (list of the future matrix values) to a matrix(!) .csv file
    # utils.run_func_for_all(
    #     func_to_run=data_processing.raw_mtx_to_csv,
    #     # folder_in_path='./raw_data',
    #     folder_in_path='./raw_data_for_tests',
    #     # folder_out_path='./raw_csv_data',
    #     folder_out_path='./raw_csv_data_for_tests',
    #     which_files=lambda x: '.mtx' in x,
    #     rename_out_files=lambda x: x[:-4] + '.csv'
    # )

    # #step 3: create the metadata files for each matrix .csv file
    # data_processing.prepare_metadata_single_files(folder_path='./raw_data_for_tests', out_folder_path='./raw_csv_data_for_tests')

    # #step 4: filter both matrices and metadatas of each sample seperately  
    # utils.run_func_for_all(
    #     func_to_run=data_processing.filter_cols,
    #     # folder_in_path='./raw_csv_data',
    #     # folder_out_path='./filtered_mtx',
    #     folder_in_path='./raw_csv_data_for_tests',
    #     folder_out_path='./raw_csv_data_for_tests',
    #     which_files=lambda x: '_matrix.csv' in x,
    #     rename_out_files= lambda x: x[:-4] + '_filtered.csv'
    # )
    # data_processing.filter_metadata_rows(folder_path='./raw_csv_data_for_tests', out_folder_path='./raw_csv_data_for_tests')
    # #plots for step 4
    # data_plot_utils.print_hist_mul("./filtered_csv_matrix_for_tests")
    # data_plot_utils.print_hist_mt_percentage(features_folder_path = './raw_csv_data_for_tests',folder_path = './raw_csv_data_for_tests')
    
    #step 5: normalize the data ti make the samples' euclidian distance 20,000
    # utils.run_func_for_all(
    #     func_to_run=data_processing.normalize_data,
    #     # folder_in_path='./filtered_mtx',
    #     folder_in_path='./raw_csv_data_for_tests',
    #     folder_out_path='./raw_csv_data_for_tests',
    #     which_files=lambda x: 'matrix_filtered.csv' in x,
    # )

    # utils.stack_csv_together('./raw_csv_data_for_tests', './raw_csv_data_for_tests/stacked_mtx.csv')
    # utils.merge_all_metadata(folder_path='./raw_csv_data_for_tests', out_folder_path='./raw_csv_data_for_tests')
    data_processing.calc_and_plot_cv('./raw_csv_data_for_tests/stacked_mtx.csv', path_to_features_csv='./raw_csv_data_for_tests/features.csv')
    pass


if __name__ == '__main__':
    main()

