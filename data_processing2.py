import pandas as pd
import numpy as np
import seaborn as sns
import os
import datetime
import matplotlib.pyplot as plt
import data_plot_utils
import utils
import time
from distinctipy import distinctipy
from colour import Color



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


def sanity_checks_gaba(path_in, path_to_MEA = './raw_data/MEA_dimorphism_samples.xlsx',
                       print_noise=True, plots_folder='./plots_folder1/part3'):
    utils.write_log('start sanity_checks_gaba')
    df_f_m_index = pd.read_excel(path_to_MEA, index_col=0, header=0)

    df = pd.read_csv(path_in, index_col=0, header=0)
    df = df.T
    if print_noise == False:
        df = df[df['dbscan_labels'] != -1]

    hist_group = {}
    hist_count = {}
    hist_tsne1 = {}
    hist_tsne2 = {}
    for index, row in df.iterrows():
        cell_id = index.split('__')[1]
        gender = df_f_m_index.at[cell_id, 'female']
        parent = df_f_m_index.at[cell_id, 'parent']
        label = int(row['dbscan_labels'])
        if label not in hist_count:
            hist_count[label] = 0
            # male_no_parent=0, male_parent=1, female_no_parent=2, female_parent=3
            hist_group[label] = [0, 0, 0, 0]
            hist_tsne1[label] = 0
            hist_tsne2[label] = 0

        hist_count[label] += 1
        hist_group[label][(2*gender) + parent] += 1
        hist_tsne1[label] += row['tsne-2d-one']
        hist_tsne2[label] += row['tsne-2d-two']

    for label in hist_group:
        hist_tsne1[label] = hist_tsne1[label]/hist_count[label]
        hist_tsne2[label] = hist_tsne2[label]/hist_count[label]

    tmp_colors = distinctipy.get_colors(len(hist_group), pastel_factor=0.6)
    gradient_colors = list(Color("red").range_to(Color("green"), 100))

    def plot_tmp():
        sns.scatterplot(
            x="tsne-2d-one",
            y="tsne-2d-two",
            hue="dbscan_labels",
            palette=tmp_colors,
            data=df,
            legend=False,
            alpha=0.3
        )
        fig = plt.gcf()
        fig.set_size_inches(16, 14)

    # male_no_parent=0, male_parent=1, female_no_parent=2, female_parent=3
    plot_tmp()
    for label in hist_group:
        score = (hist_group[label][2] + hist_group[label][3]) * 100 // hist_count[label]
        msg = f'{label}\n{score}%'
        plt.text(hist_tsne1[label], hist_tsne2[label], msg, horizontalalignment='center', size='large',
                 color=gradient_colors[score].get_rgb(), weight='bold')
    plt.title(f"DBSAN clustering (2D tSNE) on Gaba cells\nFemale Percentage")
    # data_plot_utils.save_plots(plt, f'{plots_folder}/????')
    plt.show()

    plot_tmp()
    for label in hist_group:
        score = (hist_group[label][0] + hist_group[label][1]) * 100 // hist_count[label]
        msg = f'{label}\n{score}%'
        plt.text(hist_tsne1[label], hist_tsne2[label], msg, horizontalalignment='center', size='large',
                 color=gradient_colors[score].get_rgb(), weight='bold')
    plt.title(f"DBSAN clustering (2D tSNE) on Gaba cells\nMale Percentage")
    # data_plot_utils.save_plots(plt, f'{plots_folder}/???')
    plt.show()

    plot_tmp()
    for label in hist_group:
        score = (hist_group[label][1] + hist_group[label][3]) * 100 // hist_count[label]
        msg = f'{label}\n{score}%'
        plt.text(hist_tsne1[label], hist_tsne2[label], msg, horizontalalignment='center', size='large',
                 color=gradient_colors[score].get_rgb(), weight='bold')
    plt.title(f"DBSAN clustering (2D tSNE) on Gaba cells\nParent Percentage")
    # data_plot_utils.save_plots(plt, f'{plots_folder}/???')
    plt.show()

    plot_tmp()
    for label in hist_group:
        score = (hist_group[label][3]) * 100 // hist_count[label]
        msg = f'{label}\n{score}%'
        plt.text(hist_tsne1[label], hist_tsne2[label], msg, horizontalalignment='center', size='large',
                 color=gradient_colors[score].get_rgb(), weight='bold')
    plt.title(f"DBSAN clustering (2D tSNE) on Gaba cells\nFemale Parent Percentage")
    # data_plot_utils.save_plots(plt, f'{plots_folder}/???')
    plt.show()


