import pandas as pd
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt


def raw_mtx_to_csv(file_path, path_out):
    print(f'status: start processing {file_path}')
    f_input = open(file_path, 'r')
    f_input.readline()
    f_input.readline()
    line = f_input.readline()
    col_size = int(line.split(' ')[1])
    table = np.zeros((27999, col_size + 1), dtype=int)
    for index, line in enumerate(f_input):
        if len(line) > 3:  # only read valid lines
            # print(line)
            curr_feature, curr_barcode, curr_value = line[:-1].split(' ')  # assuming already know the format
            curr_feature, curr_barcode, curr_value = int(curr_feature), int(curr_barcode), int(curr_value)
            table[curr_feature, curr_barcode] = curr_value

            if index % 1000000 == 0:
                print(f'status: reached line #{index}')

    f_input.close()
    print(f'status: finish processing mtx file. creating csv file')
    df = pd.DataFrame(table)
    df.drop([0], inplace=True, axis=0)
    df.drop([0], inplace=True, axis=1)
    df.to_csv(path_out)
    print(f'status: created the file "{path_out}"')


def filter_cols(path_in_file, path_out_file, min_sum_for_col=3000, min_diff_for_col=2500):
    print(f'status: start filtering {path_in_file} by col sum < {min_sum_for_col} and col number of different gens < '
          f'{min_diff_for_col}')
    df = pd.read_csv(path_in_file, index_col=0, header=0, dtype=np.int32)
    num_col_start = df.shape[1]
    df = df.loc[:, (df.sum(numeric_only=True) >= min_sum_for_col)]  # filter cols with sum less than 3000
    df = df.loc[:, ((df != 0).sum() > min_diff_for_col)]  # filter cols with less than different 2500 different gens
    df.to_csv(path_out_file, sep=',')
    num_col_end = df.shape[1]
    msg = f'Note: started with {num_col_start} cols, after filtering left with {num_col_end} (filtered ' \
          f'{num_col_start-num_col_end} cols)'
    print(msg)
    f = open(f'./ml_run_logs.txt', 'a+')
    msg = str(datetime.datetime.now()) + " filter_by_min_sum: " + path_in_file + ": " + msg + "\n"
    f.write(msg)
    print(f'status: finish filtering {path_in_file}. result saved to {path_out_file}')


def normalize_data(path_in_file, path_out_file, alpha=20000):
    df = pd.read_csv(path_in_file, index_col=0, header=0)

    # print(np.linalg.norm(df, axis=0))
    # print(len(np.linalg.norm(df, axis=0)))
    df = np.ceil(alpha*df/np.linalg.norm(df, axis=0))

    print(f'status: finish normalizing {path_in_file}. result saved to {path_out_file}')
    df.to_csv(path_out_file, sep=',')


def calc_and_plot_cv(path_in_file):
    df = pd.read_csv(path_in_file, index_col=0, header=0)
    print(df.shape)
    # cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100  # TODO check this formula
    cv_res = df.apply(lambda x: np.std(x, ddof=1) / np.mean(x) * 100, axis=1)  # TODO check this formula
    print("cv results:")
    print(type(cv_res))
    print(cv_res)
    cv_res = cv_res.dropna()
    print("cv results after drop NaN:")
    print(cv_res)
    cv_res_df = cv_res.to_frame()
    print("cv results as DataFrame:")
    print(cv_res_df)

    cv_res_df.plot(style="o", ms=2)
    plt.show()


