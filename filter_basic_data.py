
import pandas as pd
import numpy as np
from sklearn import preprocessing
import os


def normilize_data(path_in_file, path_out_file):
    df = pd.read_csv(path_in_file, index_col=0, header=0, dtype=np.int32)
    
    # df.
    # df_norm = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)

    df.to_csv(path_out_file, sep=',')


def filter_by_min_sum(path_in_file, path_out_file, min_value=3000):
    df = pd.read_csv(path_in_file, index_col=0, header=0, dtype=np.int32)
    print(df.head(3))
    df = df.loc[:, (df.sum(numeric_only=True) >= min_value)]
    print(df.head(3))

    df.to_csv(path_out_file, sep=',')


def filter_all_by_min_sum(folder_path="./raw_data", path_out_folder="./parsed_data"):
    raw_files = os.listdir(folder_path)  # list all raw files
    raw_files = list(filter(lambda x: '_matrix2.csv' in x, raw_files))

    for mtx in raw_files:
        sample_num = mtx[-15:-11]
        path_out_file = path_out_folder + '/' + sample_num + "_filterd_matrix.csv"
        filter_by_min_sum(mtx, path_out_file)


if __name__ == '__main__':
    filter_by_min_sum('./parsed_data/tmp22.csv', './parsed_data/tmp33_filtered.csv')

    print('Done')
    
