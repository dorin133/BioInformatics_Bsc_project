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


def print_pdf_mul(folder_path='./filtered_data3', plots_folder='./plots_folder1'):
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

def print_hist_genes(folder_path='./csv_data2', plots_folder='./plots_folder1'):
    raw_files = os.listdir(folder_path)
    raw_files = list(filter(lambda x: 'matrix.csv' in x, raw_files))
    labels = []
    plt.figure(figsize=[12, 9])
    i = 0
    print("start")
    for mtx in raw_files:
        df = pd.read_csv(folder_path + "/" + mtx, index_col=0, header=0)
        cols = df.columns
        plot_values = df[cols].gt(0).sum(axis=1)
        plt.plot(plot_values.index, plot_values.values)
        labels.append(mtx)
        print("iteration number: "+str(i))
        i+=1
        del df
    plt.legend(labels)
    plt.title("Histogram of gene expression")
    plt.xlabel("Genes' Index in features.csv")
    plt.ylabel("Number of cells with the gene's expression")
    plt.savefig(f'{plots_folder}/hist_genes{str(datetime.datetime.now().time())[:8].replace(":", "_")}.png')
    plt.show()



def plot_female_vs_male_mean(females_path, males_path, path_to_features_csv, path_stacked_mtx_file, path_out,
                             plots_folder='./plots_folder1'):
    df_f = pd.read_csv(females_path, index_col=0, header=0)
    df_m = pd.read_csv(males_path, index_col=0, header=0)

    # mean_f = pd.DataFrame(df_f.mean(axis=1))
    mean_f = np.log2(df_f.mean(axis=1)+1)
    # print("mean f\n", mean_f)
    mean_m = np.log2(df_m.mean(axis=1)+1)
    # print("mean m\n", mean_m)

    del df_f
    del df_m

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(mean_f, mean_m, s=5, c='b', marker="s")
    # plt.ax
    mean_f_np = mean_f.to_numpy()
    mean_m_np = mean_m.to_numpy()
    p = np.polyfit(mean_f_np, mean_m_np, 1)
    a, b = p
    plt.plot(mean_f_np, a*mean_f_np+b, c='r')


    # n_zeros_f = np.count_nonzero(mean_f_np==0)
    # non_zeros_f = np.count_nonzero(mean_f_np!=0)

    # print("There are "+ str(n_zeros_f)+" genes not expressed in females and "+str(non_zeros_f)+" genes expressed in females")


    # find the 20 farthest genes from p
    # think of changing this to squared distance
    dist_f = abs(mean_m_np - (a*mean_f_np+b))
    # print("dist_f", dist_f)
    dist_idx = np.argsort(dist_f)[-20:]

    print("dist_idx", dist_idx)
    df_features = pd.read_csv(path_to_features_csv, index_col=0, header=0)
    # real_idx = []
    # # print(mean_f.index)
    # for index, j in enumerate(mean_f.index):
    #     if index in dist_idx:
    #         real_idx.append(j)

    labels = df_features.iloc[dist_idx].geneName.unique()
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

    # df_all = pd.read_csv(path_stacked_mtx_file, index_col=0, header=0)  # TODO add this if needed
    # drop_idx = []
    # for index, row in df_features.iterrows():  # index start from 1
    #     if row['geneName'] in labels and index in df_all.index:
    #         drop_idx.append(index)
    # df_all = df_all.drop(drop_idx)
    # df_all.to_csv(path_out, sep=',')

    msg = f' plot_female_vs_male_mean: the 20 genes we found in this function are: {labels}'
    print(f'Status:{msg}')
    f = open(f'./ml_run_logs.txt', 'a+')
    f.write(str(datetime.datetime.now()) + msg + '\n')
    f.close()
