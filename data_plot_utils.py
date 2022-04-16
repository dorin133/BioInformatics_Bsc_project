import datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import utils


def print_hist_mt_percentage(features_folder_path='./csv_data2', folder_path='./filtered_data3', plots_folder='./plots_folder1'):
    gene_indeces = utils.find_indices_of_gene(features_folder_path)
    raw_files = os.listdir(folder_path)
    raw_files = list(filter(lambda x: 'matrix_filtered.csv' in x, raw_files))
    labels = []
    for mtx in raw_files:
        df = pd.read_csv(folder_path + "/" + mtx, index_col=0, header=0)
        sns.distplot((df.loc[gene_indeces]).sum()/df.sum(), hist=False)
        labels.append(mtx)
        del df
    # plt.xlim(left=2000)
    plt.legend(labels)
    plt.title("PDF of mitochondrial genes expression ratio per sample")
    plt.xlabel("mitochondrial genes ratio")
    plt.ylabel("probability")
    plt.savefig(f'{plots_folder}/hist_mt_percentage{str(datetime.datetime.now().time())[:8].replace(":", "_")}.png')
    plt.show()


def print_hist_mul(folder_path='./filtered_data3', plots_folder='./plots_folder1'):
    raw_files = os.listdir(folder_path)
    raw_files = list(filter(lambda x: 'matrix_filtered.csv' in x, raw_files))

    labels = []
    for mtx in raw_files:
        df = pd.read_csv(folder_path + "/" + mtx, index_col=0, header=0)
        sns.distplot(df.sum(), hist=False)
        labels.append(mtx)

        del df
    plt.xlim(left=2000)
    plt.legend(labels)
    plt.title("PDF of molecules per sample")
    plt.xlabel("molecules number")
    plt.ylabel("probability")
    plt.savefig(f'{plots_folder}/hist_mul{str(datetime.datetime.now().time())[:8].replace(":", "_")}.png')
    plt.show()

