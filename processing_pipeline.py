import utils
import data_processing
import data_plot_utils


def main():
    # utils.run_func_for_all(
    #     func_to_run=data_processing.raw_mtx_to_csv,
    #     folder_in_path='./raw_data',
    #     folder_out_path='./raw_csv_data',
    #     which_files=lambda x: '.mtx' in x,
    #     rename_out_files=lambda x: x[:-4] + '.csv'
    # )
    #
    # utils.run_func_for_all(
    #     func_to_run=data_processing.filter_cols,
    #     folder_in_path='./raw_csv_data',
    #     folder_out_path='./filtered_mtx',
    #     which_files=lambda x: '.csv' in x,
    # )
    #
    # data_plot_utils.print_hist_mul("./filtered_mtx")
    #
    # utils.run_func_for_all(
    #     func_to_run=data_processing.normalize_data,
    #     folder_in_path='./filtered_mtx',
    #     folder_out_path='./normed_mtx',
    #     which_files=lambda x: '.csv' in x,
    # )
    #
    # utils.stack_csv_together('./normed_mtx', './merged_data/stacked_mtx.csv')

    data_processing.calc_and_plot_cv('./merged_data/stacked_mtx.csv')





if __name__ == '__main__':
    main()

