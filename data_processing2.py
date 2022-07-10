import pandas as pd
import numpy as np
import seaborn as sns
import os
import datetime
import matplotlib.pyplot as plt
import data_plot_utils
import utils
import time


def filter_gaba_only(path_clust_labels, folder_path_in, folder_path_out):
    utils.write_log(f'start filter_gaba_only')
    cell_to_label = pd.read_csv(path_clust_labels, index_col=0, header=0).T

    raw_files = os.listdir(folder_path_in)  # list all raw files
    chosen_files = list(filter(lambda x: 'matrix.csv' in x, raw_files))
    chosen_files.sort()
    # print(f'gonna work with the following files: {chosen_files}')
    for file_name in chosen_files:
        print(f'working with file: {file_name}')
        df = pd.read_csv(f'{folder_path_in}/{file_name}', index_col=0, header=0)
        df_id = file_name[:-11]  # remove '_matrix.csv'
        cols = df.columns
        # print(f'cols: {len(cols)}: {cols}')
        keep_cols = []
        for col_name in cols:
            tmp = f'{col_name}__{df_id}'
            if tmp in cell_to_label.index and cell_to_label.at[tmp, 'nueral_labels'] == 'Gaba':
                keep_cols.append(col_name)
        # print(f'keep_cols: {len(keep_cols)}: {keep_cols}')

        msg = f'{df_id}: Original df shape is {df.shape}.'
        df = df[keep_cols]
        msg += f' Only gaba df shape is {df.shape}.'
        utils.write_log(msg)

        path_out_file_with_name = f'{folder_path_out}/{df_id}_gaba_matrix.csv'
        df.to_csv(path_out_file_with_name, sep=',')
        del df

    utils.write_log(f'end filter_gaba_only_wrapper')


def filter_rare_gens(path_stacked_mtx_file, path_out_file):
    utils.write_log(f'start filter_rare_gens')
    df = pd.read_csv(path_stacked_mtx_file, index_col=0, header=0)
    original_shape = df.shape
    hist_row_non_zeros = (df != 0).sum(axis=1)
    df_filtered = df[5 < hist_row_non_zeros]
    hist_row_non_zeros = (df_filtered != 0).sum(axis=1)
    print('df.shape', df.shape)
    print('df.shape[0]', df.shape[0])
    print('df.shape[1]', df.shape[1])
    # df_filtered = df_filtered[hist_row_non_zeros < df.shape[1] / 2]  # TODO
    utils.write_log(f'filtered {df.shape[0]-df_filtered.shape[0]} genes (original shape was {original_shape} and the '
                    f'update one is {df_filtered.shape}). filtered csv saved as {path_out_file}')
    df_filtered.to_csv(path_out_file, sep=',')

