import utils
import data_processing
import data_plot_utils
import ml_processing
import pandas as pd
import linkage_and_heatmap as link_and_heat
import os
import gaba_genes_processing
import data_processing2
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
import data_plot_utils
import time
from distinctipy import distinctipy
from colour import Color
import cv2
from matplotlib import cycler
IPython_default = plt.rcParams.copy()

def barplot_dataset(path_to_MEA='./raw_data/MEA_dimorphism_samples.xlsx', plots_folder= './plots_folder1/report_missing_plots'):
    utils.write_log(f'start clusters_bar_groups')
    colors = cycler('color',
                    ['#0B5394', '#3388BB', '#9988DD',
                     '#EECC55', '#88BB44', '#FFBBBB'])
    plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
           axisbelow=True, grid=True, prop_cycle=colors)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('xtick', direction='out', color='gray')
    plt.rc('ytick', direction='out', color='gray')
    plt.rc('patch', edgecolor='#E6E6E6')
    plt.rc('lines', linewidth=2)
    df_f_m_index = pd.read_excel(path_to_MEA, index_col=0, header=0)
    # cell_to_label = pd.read_csv(path_in, index_col=0, header=0).T
    hist_group = {0: 0, 1: 0, 2: 0, 3: 0}
    for index, row in df_f_m_index.iterrows():
        # cell_id = index.split('__')[1]
        gender = df_f_m_index.at[index, 'female']
        parent = df_f_m_index.at[index, 'parent']
        # male_no_parent=0, male_parent=1, female_no_parent=2, female_parent=3
        hist_group[(2 * gender) + parent] += 1
    # if -1 in hist_group:  # remove from comment if noise cluster should be ignored
    #     del hist_group[-1]
    hist_df = pd.DataFrame.from_dict(hist_group, orient='index').T
    # hist_df = hist_df.div(hist_df.sum(axis=1), axis=0)
    # male_no_parent=0, male_parent=1, female_no_parent=2, female_parent=3
    hist_df.columns = ['naive male', 'parent and male', 'naive female', 'female and parent']
    hist_df = hist_df.T
    hist_df.sort_index(axis=0, inplace=True)
    ax = hist_df.plot.bar(stacked=True, figsize=(16, 10), rot=0)
    plt.title('Dataset Distribution Over: Males, Females, Parent and Naive mice groups', fontdict={'fontsize': 25}, pad=30)
    ax.tick_params(axis='x', which='both', labelsize=18)
    ax.tick_params(axis='y', which='both', labelsize=18)
    ax.legend(["amount"], fontsize=20)
    ax.set_ylim(0, 5)
    data_plot_utils.save_plots(plt, f'{plots_folder}/clusters_bar_groups')
    plt.show()
    pass




if __name__ == '__main__':
    barplot_dataset()
    print("Done")