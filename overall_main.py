import data_to_matrix
import filter_basic_data

if __name__ == '__main__':
    data_to_matrix.all_mtx_to_pandas('./raw_data', './raw_data2')
    filter_basic_data.filter_all_by_min_sum('./raw_data2', './raw_data3')
    data_to_matrix.stack_all_csv_together('./raw_data3', './parsed_data/all_parsed_data.csv')
