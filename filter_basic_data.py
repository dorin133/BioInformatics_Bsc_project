
import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
import matplotlib.pyplot as plt
import seaborn as sns



# def filter_and_normalize(path_in_file, path_out_file, min_value=3000, alpha=20000, flag_round=True):
#     print(f'status: start filtering {path_in_file} by col sum < {min_value}')
#     df = pd.read_csv(path_in_file, index_col=0, header=0, dtype=np.int32)
#     df = df.loc[:, (df.sum(numeric_only=True) >= min_value)]
#     print(f'status: start normalize {path_in_file} by with alpha={alpha}')
#     if flag_round:
#         df=round(alpha*df/df.sum())
#     else:
#         df=alpha*df/df.sum()
#     print(f'status: finish filtering and normalizing {path_in_file}. result saved to {path_out_file}')
#     df.to_csv(path_out_file, sep=',')


# def filter_and_norm_all(folder_path="./raw_data", path_out_folder="./parsed_data"):
#     raw_files = os.listdir(folder_path)  # list all raw files
#     raw_files = list(filter(lambda x: '_matrix2.csv' in x, raw_files))

#     for mtx in raw_files:
#         sample_num = mtx[-16:-12]
#         path_out_file = path_out_folder + '/' + sample_num + "_filterd_normed_matrix2.csv"
#         filter_and_normalize(folder_path + "/" + mtx, path_out_file)


def normalize_data(path_in_file, path_out_file, alpha=20000, flag_round=True):
    df = pd.read_csv(path_in_file, index_col=0, header=0, dtype=np.float128)
    
    normalized_df= alpha*df/df.sum()
    normalized_df.to_csv(path_out_file, sep=',')

    if flag_round:
        df=round(alpha*df/df.sum())
    else:
        df=alpha*df/df.sum()
    print(f'status: finish normalizing {path_in_file}. result saved to {path_out_file}')
    df.to_csv(path_out_file, sep=',')



def normilize_all(folder_path="./raw_data", path_out_folder="./parsed_data"):
    raw_files = os.listdir(folder_path)  # list all raw files
    raw_files = list(filter(lambda x: '_matrix2.csv' in x, raw_files))

    for mtx in raw_files:
        sample_num = mtx[-16:-12]
        path_out_file = path_out_folder + '/' + sample_num + "_norm_matrix.csv"
        normalize_data(folder_path + "/" + mtx, path_out_file)


def filter_by_min_sum(path_in_file, path_out_file, min_value=3000):
    print(f'status: start filtering {path_in_file} by col sum < {min_value}')
    df = pd.read_csv(path_in_file, index_col=0, header=0, dtype=np.int32)
    df = df.loc[:, (df.sum(numeric_only=True) >= min_value)]
    df.to_csv(path_out_file, sep=',')
    print(f'status: finish filtering {path_in_file}. result saved to {path_out_file}')


def filter_all_by_min_sum(folder_path="./raw_data", path_out_folder="./parsed_data"):
    raw_files = os.listdir(folder_path)  # list all raw files
    raw_files = list(filter(lambda x: '_matrix2.csv' in x, raw_files))

    for mtx in raw_files:
        sample_num = mtx[-16:-12]
        path_out_file = path_out_folder + '/' + sample_num + "_filterd_matrix.csv"
        filter_by_min_sum(folder_path + "/" + mtx, path_out_file)


def print_hist_mul(folder_path):
    raw_files = os.listdir(folder_path) 
    raw_files = list(filter(lambda x: '_filterd_matrix' in x, raw_files))

    labels = []
    for mtx in raw_files:
        df = pd.read_csv(folder_path + "/" + mtx, index_col=0, header=0, dtype=np.int32)
        sns.distplot(df.sum(), hist=False)
        labels.append(mtx)

        del df
    plt.xlim(left = 2000)
    plt.legend(labels)
    plt.title("PDF of molecules per sample")
    plt.xlabel("molecules number")
    plt.ylabel("probability")
    plt.show()


def print_hist_gens(folder_path):
    raw_files = os.listdir(folder_path) 
    raw_files = list(filter(lambda x: '_filterd_matrix' in x, raw_files))

    labels = []
    for mtx in raw_files:
        df = pd.read_csv(folder_path + "/" + mtx, index_col=0, header=0, dtype=np.int32)
        sns.distplot(df.sum(axis=1), hist=False)
        labels.append(mtx)

        del df
    plt.legend(labels)
    plt.title("PDF of genes per sample")
    plt.xlabel("gene")
    plt.ylabel("probability")
    plt.show()


def print_hist_gens2(folder_path):
    raw_files = os.listdir(folder_path) 
    raw_files = list(filter(lambda x: '_filterd_matrix' in x, raw_files))

    for mtx in raw_files:
        df = pd.read_csv(folder_path + "/" + mtx, index_col=0, header=0, dtype=np.int32)
        df.drop(df[df.sum(axis=1) == 0].index, inplace=True)
        sns.distplot(df.sum(axis=1), hist=False)
        del df
        plt.title(f"PDF of genes per sample {mtx}")
        plt.xlabel("gene")
        plt.ylabel("probability")
        plt.show()


if __name__ == '__main__':
    # filter_by_min_sum('./parsed_data/tmp22.csv', './parsed_data/tmp33_filtered.csv')
    # filter_all_by_min_sum('./raw_data2', './parsed_data')
    # print_hist_mul('./raw_data3')
    # print_hist_gens('./raw_data3')
    print_hist_gens2('./raw_data3')

    print('Done')
    
