import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def print_hist_mt_percentage(folder_path = './filtered_mtx'):
    raw_files = os.listdir(folder_path)
    raw_files = list(filter(lambda x: '.csv' in x, raw_files))

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
    plt.show()


def print_hist_mul(folder_path = './filtered_mtx'):
    raw_files = os.listdir(folder_path)
    raw_files = list(filter(lambda x: '.csv' in x, raw_files))

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
    plt.show()
