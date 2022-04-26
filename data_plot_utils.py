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


def plot_female_vs_male_mean(females_path, males_path, path_to_features_csv='./csv_data2/features.csv', plots_folder=
'./plots_folder1'):
    df_f = pd.read_csv(females_path, index_col=0, header=0)
    df_m = pd.read_csv(males_path, index_col=0, header=0)

    # mean_f = pd.DataFrame(df_f.mean(axis=1))
    mean_f = df_f.mean(axis=1)
    print("mean f\n", mean_f)
    mean_m = df_m.mean(axis=1)
    print("mean m\n", mean_m)


    # print(mean_f)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(mean_f, mean_m, s=5, c='b', marker="s")
    # plt.ax
    p = np.polyfit(mean_f.to_numpy(), mean_m.to_numpy(), 1)
    a, b = p
    plt.plot(mean_f, a*mean_f+b, c='r')

    mean_f_np = mean_f.to_numpy()
    mean_m_np = mean_m.to_numpy()

    # find the 20 farthest genes from p
    dist_f = abs(mean_m_np - (a*mean_f_np+b))
    # print("dist_f", dist_f)
    dist_idx = np.argsort(dist_f)[-20:]
    print("dist_idx", dist_idx)
    df_features = pd.read_csv(path_to_features_csv, index_col=0, header=0)
    labels = df_features.loc[dist_idx+1].geneName.unique()  # notice the "+1" to fix the diff between the two
    i = 0

    # add to the plot the names of the farthest genes
    for x, y in zip(mean_f_np[dist_idx], mean_m_np[dist_idx]):
        plt.annotate(labels[i],  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 2),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
        i += 1
    plt.title("Females vs Males genes mean")
    plt.ylabel("Males")  # TODO double check this
    plt.xlabel("Females")
    plt.savefig(f'{plots_folder}/female_vs_male_mean{str(datetime.datetime.now().time())[:8].replace(":", "_")}.png')
    plt.show()

    # df_threshold = pd.read_csv(<all_merged_file_path>, index_col=0, header=0)  # TODO add this if needed
    # manually_remove = labels
    # print(f'Also manually removing {manually_remove}')
    # drop_idx = []
    # for index, row in df_features.iterrows():  # index start from 1
    #     if row['geneName'] in manually_remove and index in df_threshold.index:
    #         drop_idx.append(index)
    # df_threshold = df_threshold.drop(drop_idx)

    msg = f' plot_female_vs_male_mean: the 20 genes we found in this function are: {labels}'
    print(f'Status:{msg}')
    f = open(f'./ml_run_logs.txt', 'a+')
    f.write(str(datetime.datetime.now()) + msg + '\n')
    f.close()
