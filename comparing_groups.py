import utils
import plots_folder1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import ranksums
from math import log10
from matplotlib.patches import Ellipse

import data_plot_utils
import os


def split_clusters_and_groups(path_in_data, path_in_groups_data, folder_out):
    utils.write_log(f'start split_clusters_and_groups')
    cell_to_label = pd.read_csv(path_in_groups_data, index_col=0, header=0).T

    clusters = cell_to_label['linkage_labels'].unique().tolist()
    if -1 in clusters:
        clusters.remove(-1)
    clusters.sort()
    print(f'{len(clusters)} clusters:', clusters)
    all_data = pd.read_csv(path_in_data, index_col=0, header=0)
    for cluster in clusters:
        print(f'cluster {cluster}')
        cluster_cells_ids = cell_to_label[cell_to_label['linkage_labels'] == cluster].index.to_list()
        cluster_cells_data_all = all_data[cluster_cells_ids]
        cluster_cells_data_all.to_csv(f'{folder_out}/{cluster}_all.csv')

        cluster_cells_ids = cell_to_label[
            (cell_to_label['linkage_labels'] == cluster) & (cell_to_label['female'] == '0')].index.to_list()
        cluster_cells_data = cluster_cells_data_all[cluster_cells_ids]
        cluster_cells_data.to_csv(f'{folder_out}/{cluster}_males.csv')

        cluster_cells_ids = cell_to_label[
            (cell_to_label['linkage_labels'] == cluster) & (cell_to_label['female'] == '1')].index.to_list()
        cluster_cells_data = cluster_cells_data_all[cluster_cells_ids]
        cluster_cells_data.to_csv(f'{folder_out}/{cluster}_females.csv')

        cluster_cells_ids = cell_to_label[
            (cell_to_label['linkage_labels'] == cluster) & (cell_to_label['parent'] == '0')].index.to_list()
        cluster_cells_data = cluster_cells_data_all[cluster_cells_ids]
        cluster_cells_data.to_csv(f'{folder_out}/{cluster}_virgins.csv')

        cluster_cells_ids = cell_to_label[
            (cell_to_label['linkage_labels'] == cluster) & (cell_to_label['parent'] == '1')].index.to_list()
        cluster_cells_data = cluster_cells_data_all[cluster_cells_ids]
        cluster_cells_data.to_csv(f'{folder_out}/{cluster}_parents.csv')

        cluster_cells_ids = cell_to_label[
            (cell_to_label['linkage_labels'] == cluster) & (cell_to_label['parent'] == '1') & (
                        cell_to_label['female'] == '0')].index.to_list()
        cluster_cells_data = cluster_cells_data_all[cluster_cells_ids]
        cluster_cells_data.to_csv(f'{folder_out}/{cluster}_male_parents.csv')

        cluster_cells_ids = cell_to_label[
            (cell_to_label['linkage_labels'] == cluster) & (cell_to_label['parent'] == '1') & (
                        cell_to_label['female'] == '1')].index.to_list()
        cluster_cells_data = cluster_cells_data_all[cluster_cells_ids]
        cluster_cells_data.to_csv(f'{folder_out}/{cluster}_female_parents.csv')

        cluster_cells_ids = cell_to_label[
            (cell_to_label['linkage_labels'] == cluster) & (cell_to_label['parent'] == '0') & (
                        cell_to_label['female'] == '0')].index.to_list()
        cluster_cells_data = cluster_cells_data_all[cluster_cells_ids]
        cluster_cells_data.to_csv(f'{folder_out}/{cluster}_male_virgins.csv')

        cluster_cells_ids = cell_to_label[
            (cell_to_label['linkage_labels'] == cluster) & (cell_to_label['parent'] == '0') & (
                        cell_to_label['female'] == '1')].index.to_list()
        cluster_cells_data = cluster_cells_data_all[cluster_cells_ids]
        cluster_cells_data.to_csv(f'{folder_out}/{cluster}_female_virgins.csv')

    utils.write_log(f'end split_clusters_and_groups')


def compere_female_vs_male_for_each_cluster(folder_in, path_to_features_csv='./csv_data2/features.csv',
                                            plots_folder='./plots_folder1/part4'):

    raw_files = os.listdir(folder_in)  # list all raw files
    tmp_files = list(filter(lambda x: '_all.csv' in x, raw_files))
    clusters = [x[:-8] for x in tmp_files]
    clusters.sort()
    print(clusters)
    # for cluster in clusters:  # TODO
    for cluster in clusters[:6]:
        print(f'cluster {cluster}')
        for group_type in ['parents', 'virgins']:
            female_file = f'{folder_in}/{cluster}_female_{group_type}.csv'
            male_file = f'{folder_in}/{cluster}_male_{group_type}.csv'

        data_plot_utils.plot_female_vs_male_mean(females_path=female_file,
                                                 males_path=male_file,
                                                 path_to_features_csv=path_to_features_csv,
                                                 path_stacked_mtx_file='?????Not yet imp?????',  # TODO
                                                 path_out='????Not yet imp?????',
                                                 plots_folder=plots_folder,
                                                 append_to_plot_name=f'{int(cluster)}_{group_type}')

        data_plot_utils.plot_female_vs_male_fraction_expression(females_path=female_file,
                                                                males_path=male_file,
                                                                path_to_features_csv=path_to_features_csv,
                                                                path_stacked_mtx_file='?????Not yet imp?????',  # TODO
                                                                path_out='????Not yet imp?????',
                                                                plots_folder=plots_folder,
                                                                append_to_plot_name=f'{int(cluster)}_{group_type}')


def plot_female_vs_male_mean_2(females_path, males_path, path_to_features_csv,
                             plots_folder, append_to_plot_name=''):
    gr1 = pd.read_csv(females_path, index_col=0, header=0)
    gr2 = pd.read_csv(males_path, index_col=0, header=0)

    mean_f = np.log2(gr1 + 1).mean(axis=1)
    # print("mean f\n", mean_f)
    mean_m = np.log2(gr2 + 1).mean(axis=1)
    # print("mean m\n", mean_m)

    del gr1
    del gr2

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(mean_f, mean_m, s=5, c='b', marker="s")

    # p = np.poly1d(np.polyfit(mean_f, mean_m, 1))
    # plt.plot(np.unique(mean_f), p(np.unique(mean_f)))
    p = np.poly1d(np.polyfit(mean_f-mean_m, (mean_f+mean_m)/2, 1))
    plt.plot(np.unique(mean_f-mean_m), p(np.unique((mean_f+mean_m)/2)))

    # find the 20 farthest genes from p
    dist_f = abs(mean_m - p(mean_f))
    # dist_cv.index contains the real indeces of the genes (by the features table)
    dist_f_dict = dict(zip(dist_f.index, dist_f.values))
    # the next line returns the real indeces (by the features table) of the 100 genes with the highest cv
    dist_f_20 = sorted(dist_f_dict, key=dist_f_dict.get, reverse=True)[:20]
    # print("dist_f", dist_f)
    # dist_idx = np.argsort(dist_f)[-20:]

    # print("dist_idx", dist_idx)
    gr1_features = pd.read_csv(path_to_features_csv, index_col=0, header=0)

    labels = gr1_features.loc[dist_f_20].geneName.unique()
    i = 0
    # add to the plot the names of the farthest genes
    for x, y in zip(mean_f.loc[dist_f_20], mean_m.loc[dist_f_20]):
        plt.annotate(labels[i],  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 2),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
        i += 1

    if append_to_plot_name != '':
        plt.title(f"Females vs Males genes mean ({append_to_plot_name})")
        append_to_plot_name = f'_{append_to_plot_name}'
    else:
        plt.title("Females vs Males genes mean")
    plt.ylabel("mean(Males)")  # TODO double check this
    plt.xlabel("mean(Females)")
    # plt.savefig(f'{plots_folder}/female_vs_male_mean{str(datetime.datetime.now().time())[:8].replace(":", "_")}.png')
    save_plots(plt, f'{plots_folder}/female_vs_male_mean{append_to_plot_name}')
    plt.show()

    # df_all = pd.read_csv(path_stacked_mtx_file, index_col=0, header=0)  # TODO add this if needed
    # drop_idx = []
    # for index, row in gr1_features.iterrows():  # index start from 1
    #     if row['geneName'] in labels and index in df_all.index:
    #         drop_idx.append(index)
    # df_all = df_all.drop(drop_idx)
    # df_all.to_csv(path_out, sep=',')

    utils.write_log(f'plot_female_vs_male_mean: the 20 genes we found in this function are: {labels}')


def ranksum_plot(folder_in, path_to_features='./csv_data2/features.csv', plots_folder='./plots_folder1/part4'):
    utils.write_log(f'start ranksum_plot')
    raw_files = os.listdir(folder_in)  # list all raw files
    tmp_files = list(filter(lambda x: '_all.csv' in x, raw_files))
    clusters = [x[:-8] for x in tmp_files]
    clusters.sort()
    print(clusters)

    features = pd.read_csv(path_to_features, index_col=0, header=0)

    for cluster in clusters:
        print(f'cluster {cluster}')
        female_file = f'{folder_in}/{cluster}_females.csv'
        male_file = f'{folder_in}/{cluster}_males.csv'

        females = pd.read_csv(female_file, index_col=0, header=0).T
        males = pd.read_csv(male_file, index_col=0, header=0).T

        log_pvalue_res = []
        means_diff = []
        gens_names = []
        for i in males.columns:
            vec1 = females[i]
            vec2 = males[i]
            mean1 = vec1.mean()
            mean2 = vec2.mean()
            full_res = ranksums(vec1, vec2)
            # print(full_res)
            pvalue_res = full_res.pvalue
            # print(pvalue_res)
            log_pvalue_res.append(-log10(pvalue_res))
            means_diff.append(mean2 - mean1)
            gens_names.append(features.loc[i, 'geneName'])

        fig, ax = plt.subplots()
        ax.scatter(means_diff, log_pvalue_res, s=5)
        texts = []
        radius_w = (plt.xlim()[1] - plt.xlim()[0]) / 50
        radius_h = (plt.ylim()[1] - plt.ylim()[0]) / 43
        for i, txt in enumerate(gens_names):
            if log_pvalue_res[i] > 3:
                texts.append(ax.text(means_diff[i], log_pvalue_res[i], txt))
                ellipse = Ellipse(xy=(means_diff[i], log_pvalue_res[i]), width=radius_w,
                                  height=radius_h, edgecolor='r', fc='None', lw=1, fill=False)
                ax.add_patch(ellipse)
        plt.title(f"Ranksums for cluster {cluster} - females vs males")
        plt.xlabel("means_diff (males-females)")  # TODO double check this
        plt.ylabel("-log10(ranksum_pvalue)")
        # plt.savefig(f'{plots_folder}/female_vs_male_mean{str(datetime.datetime.now().time())[:8].replace(":", "_")}.png')
        data_plot_utils.save_plots(plt, f'{plots_folder}/ranksum_{cluster}')
        plt.show()
    utils.write_log(f'end ranksum_plot')


if __name__ == '__main__':
    # split_clusters_and_groups(path_in_data='./gaba_merged_data10/gaba_stacked_2_v2.csv',
    #                           path_in_groups_data='./gaba_clustered_data11/gaba_all_clustter_stats.csv',
    #                           folder_out='groups')

    # compere_female_vs_male_for_each_cluster(folder_in='groups')

    ranksum_plot(folder_in='groups')

