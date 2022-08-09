import utils
import plots_folder1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
        cluster_cells_data_all.to_csv(f'{folder_out}/{int(float(cluster))}_all.csv')

        cluster_cells_ids = cell_to_label[
            (cell_to_label['linkage_labels'] == cluster) & (cell_to_label['female'] == '0')].index.to_list()
        cluster_cells_data = cluster_cells_data_all[cluster_cells_ids]
        cluster_cells_data.to_csv(f'{folder_out}/{int(float(cluster))}_males.csv')

        cluster_cells_ids = cell_to_label[
            (cell_to_label['linkage_labels'] == cluster) & (cell_to_label['female'] == '1')].index.to_list()
        cluster_cells_data = cluster_cells_data_all[cluster_cells_ids]
        cluster_cells_data.to_csv(f'{folder_out}/{int(float(cluster))}_females.csv')

        cluster_cells_ids = cell_to_label[
            (cell_to_label['linkage_labels'] == cluster) & (cell_to_label['parent'] == '0')].index.to_list()
        cluster_cells_data = cluster_cells_data_all[cluster_cells_ids]
        cluster_cells_data.to_csv(f'{folder_out}/{int(float(cluster))}_virgins.csv')

        cluster_cells_ids = cell_to_label[
            (cell_to_label['linkage_labels'] == cluster) & (cell_to_label['parent'] == '1')].index.to_list()
        cluster_cells_data = cluster_cells_data_all[cluster_cells_ids]
        cluster_cells_data.to_csv(f'{folder_out}/{int(float(cluster))}_parents.csv')

        cluster_cells_ids = cell_to_label[
            (cell_to_label['linkage_labels'] == cluster) & (cell_to_label['parent'] == '1') & (
                        cell_to_label['female'] == '0')].index.to_list()
        cluster_cells_data = cluster_cells_data_all[cluster_cells_ids]
        cluster_cells_data.to_csv(f'{folder_out}/{int(float(cluster))}_male_parents.csv')

        cluster_cells_ids = cell_to_label[
            (cell_to_label['linkage_labels'] == cluster) & (cell_to_label['parent'] == '1') & (
                        cell_to_label['female'] == '1')].index.to_list()
        cluster_cells_data = cluster_cells_data_all[cluster_cells_ids]
        cluster_cells_data.to_csv(f'{folder_out}/{int(float(cluster))}_female_parents.csv')

        cluster_cells_ids = cell_to_label[
            (cell_to_label['linkage_labels'] == cluster) & (cell_to_label['parent'] == '0') & (
                        cell_to_label['female'] == '0')].index.to_list()
        cluster_cells_data = cluster_cells_data_all[cluster_cells_ids]
        cluster_cells_data.to_csv(f'{folder_out}/{int(float(cluster))}_male_virgins.csv')

        cluster_cells_ids = cell_to_label[
            (cell_to_label['linkage_labels'] == cluster) & (cell_to_label['parent'] == '0') & (
                        cell_to_label['female'] == '1')].index.to_list()
        cluster_cells_data = cluster_cells_data_all[cluster_cells_ids]
        cluster_cells_data.to_csv(f'{folder_out}/{int(float(cluster))}_female_virgins.csv')

    utils.write_log(f'end split_clusters_and_groups')


def compere_female_vs_male_for_each_cluster(folder_in, path_to_features_csv='./csv_data2/features.csv',
                                            plots_folder='./plots_folder1/part4'):
    utils.write_log(f'start compere_female_vs_male_for_each_cluster')
    raw_files = os.listdir(folder_in)  # list all raw files
    tmp_files = list(filter(lambda x: '_all.csv' in x, raw_files))
    clusters = [x[:-8] for x in tmp_files]
    clusters.sort()
    print(clusters)
    for cluster in clusters:
        print(f'cluster {cluster}')
        for group_type in ['parents', 'virgins']:
            female_file = f'{folder_in}/{cluster}_female_{group_type}.csv'
            male_file = f'{folder_in}/{cluster}_male_{group_type}.csv'

            plot_female_vs_male_mean_2(females_path=female_file, males_path=male_file,
                                       path_to_features_csv=path_to_features_csv,
                                       plots_folder=plots_folder,
                                       append_to_plot_name=f'{cluster}_{group_type}',
                                       frac_or_not=False)  # TODO notice the option to run as frac and as naive
    utils.write_log(f'end compere_female_vs_male_for_each_cluster')


def plot_female_vs_male_mean_2(females_path, males_path, path_to_features_csv,
                             plots_folder, append_to_plot_name='', frac_or_not=True):
    utils.write_log(f'start plot_female_vs_male_mean_2')
    gr1 = pd.read_csv(females_path, index_col=0, header=0)
    gr2 = pd.read_csv(males_path, index_col=0, header=0)

    if frac_or_not:
        mean_gr1 = gr1.astype(bool).sum(axis=1) / gr1.shape[1]
        mean_gr2 = gr2.astype(bool).sum(axis=1) / gr2.shape[1]
    else:
        mean_gr1 = np.log2(gr1 + 1).mean(axis=1)
        mean_gr2 = np.log2(gr2 + 1).mean(axis=1)

    del gr1  # just to save some memory
    del gr2

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    mean_diff = mean_gr1-mean_gr2
    avg_means = (mean_gr1+mean_gr2)/2
    ax1.scatter(avg_means, mean_diff, s=4, c='b', marker="s")

    # p = np.poly1d(np.polyfit(avg_means, mean_diff, 1))  # TODO the poly fit returns non linear graph. ask Amit about it
    # plt.plot(avg_means, p(mean_diff))
    plt.axhline(y=0, color='black', linestyle='-')
    plt.axhline(y=1, color='black', linestyle='--', linewidth=0.5)
    plt.axhline(y=-1, color='black', linestyle='--', linewidth=0.5)
    plt.grid(visible=True, which='both')

    plt.title(f"Gean expression mean vs diff with females vs males {append_to_plot_name}")
    plt.xlabel("(gr1+gr2)/2")  # TODO double check this
    plt.ylabel("gr1-gr2")

    # find the 20 farthest genes from 0
    dist_y = (mean_diff - 0)  # TODO 0 since ploy is manualy set now as 0

    for rev in [True, False]:
        dist_f_dict = dict(zip(dist_y.index, dist_y.values))
        dist_f_20 = sorted(dist_f_dict, key=dist_f_dict.get, reverse=rev)[:20]
        gr1_features = pd.read_csv(path_to_features_csv, index_col=0, header=0)
        labels = gr1_features.loc[dist_f_20].geneName.unique()
        i = 0
        # add to the plot the names of the farthest genes
        for x, y in zip(avg_means.loc[dist_f_20], mean_diff.loc[dist_f_20]):
            plt.annotate(labels[i],  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 2),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center
            plt.plot(x, y, 'ro', markersize=2)
            i += 1

    data_plot_utils.save_plots(plt, plot_name=f'{plots_folder}/mean_vs_diff_{append_to_plot_name}_')
    plt.show()
    utils.write_log(f'end plot_female_vs_male_mean_2')


def ranksum_plot(folder_in, res_out, path_to_features='./csv_data2/features.csv', plots_folder='./plots_folder1/part4'):
    utils.write_log(f'start ranksum_plot')
    raw_files = os.listdir(folder_in)  # list all raw files
    tmp_files = list(filter(lambda x: '_all.csv' in x, raw_files))
    clusters = [x[:-8] for x in tmp_files]
    clusters.sort()
    print(clusters)

    features = pd.read_csv(path_to_features, index_col=0, header=0)
    res_dict = {}
    dimo_gens = set()

    for cluster in clusters:
        print(f'cluster {cluster}')
        for group_type in ['s', '_parents', '_virgins']:
            res_dict[f'{cluster}_female{group_type}'] = set()
            res_dict[f'{cluster}_male{group_type}'] = set()
            female_file = f'{folder_in}/{cluster}_female{group_type}.csv'
            male_file = f'{folder_in}/{cluster}_male{group_type}.csv'

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
                means_diff.append(log10(mean2+1) - log10(mean1+1))  # means_diff.append(mean2 - mean1)
                gens_names.append(features.loc[i, 'geneName'])

            fig, ax = plt.subplots()
            ax.scatter(means_diff, log_pvalue_res, s=5)
            texts = []
            radius_w = (plt.xlim()[1] - plt.xlim()[0]) / 50
            radius_h = (plt.ylim()[1] - plt.ylim()[0]) / 43
            above_gender = {True: 0, False: 0}
            for i, txt in enumerate(gens_names):
                treshold = 0.2
                if log_pvalue_res[i] > 3 and (means_diff[i] > treshold or means_diff[i] < -treshold):
                    texts.append(ax.text(means_diff[i], log_pvalue_res[i], txt))
                    ellipse = Ellipse(xy=(means_diff[i], log_pvalue_res[i]), width=radius_w,
                                      height=radius_h, edgecolor='r', fc='None', lw=1, fill=False)
                    ax.add_patch(ellipse)
                    above_gender[means_diff[i] > 0] += 1

                    if means_diff[i] > treshold:
                        res_dict[f'{cluster}_male{group_type}'].add(txt)
                    else:
                        res_dict[f'{cluster}_female{group_type}'].add(txt)
                    dimo_gens.add(txt)

            plt.title(f"Ranksums for cluster {cluster} - female{group_type} vs male{group_type}")
            # plt.title(f"Ranksums for cluster {cluster} - female{group_type} vs male{group_type}\n{above_gender[True]} males above, {above_gender[False]} females above")
            utils.write_log(f"Ranksums for cluster {cluster} - female{group_type} vs male{group_type}: {above_gender[True]} males above, {above_gender[False]} females above")
            plt.xlabel(f"means_diff [log(males)-log(females)]")
            plt.ylabel("-log10(ranksum_log_pvalue)")

            data_plot_utils.save_plots(plt, f'{plots_folder}/ranksum_{cluster}_female{group_type}_vs_male{group_type}', append_time='')
            plt.show()

    data = []
    for group in res_dict:
        data.append([(1 if g in res_dict[group] else 0) for g in dimo_gens])
    df = pd.DataFrame(data, index=res_dict.keys(), columns=dimo_gens)
    df.to_csv(res_out)

    utils.write_log(f'end ranksum_plot')


def ranksum_present_results(path_in, res_out, plots_folder='./plots_folder1/part4'):
    # utils.write_log(f'start ranksum_present_results')
    df = pd.read_csv(path_in)
    df = df[df['Unnamed: 0'].str.len() > 10]
    df.set_index('Unnamed: 0', inplace=True)

    fig, ax = plt.subplots(figsize=(15, 35))
    sns.heatmap(df, cbar=False)
    # ax.format_coord = lambda x, y: 'x={:d}, y={:d}, z={}'.format(int(np.floor(x)), int(np.floor(y), ), f'{int(np.floor(x))}_{int(np.floor(y))}')

    data_plot_utils.save_plots(plt, f'{plots_folder}/ranksum1_present_results')  # TODO
    plt.show()

    male_parents = list(filter(lambda x: '_male_parents' in x, df.index.to_list()))
    female_parents = list(filter(lambda x: '_female_parents' in x, df.index.to_list()))
    male_virgins = list(filter(lambda x: '_male_virgins' in x, df.index.to_list()))
    female_virgins = list(filter(lambda x: '_female_virgins' in x, df.index.to_list()))
    data = {}
    data['male_parents'] = df.loc[male_parents].sum().tolist()
    data['female_parents'] = df.loc[female_parents].sum().tolist()
    data['male_virgins'] = df.loc[male_virgins].sum().tolist()
    data['female_virgins'] = df.loc[female_virgins].sum().tolist()

    df2 = pd.DataFrame(data, index=df.loc[male_parents].sum().index)
    utils.write_log(f"male_parents: top gens are:\n{df2.nlargest(10, 'male_parents')['male_parents']}")
    utils.write_log(f"female_parents: top gens are:\n{df2.nlargest(10, 'female_parents')['female_parents']}")
    utils.write_log(f"male_virgins: top gens are:\n{df2.nlargest(10, 'male_virgins')['male_virgins']}")
    utils.write_log(f"female_virgins: top gens are:\n{df2.nlargest(10, 'female_virgins')['female_virgins']}")
    res_dict = {}
    res_dict['male_parents'] = df2.nlargest(10, 'male_parents')['male_parents'].keys().tolist()
    res_dict['female_parents'] = df2.nlargest(10, 'female_parents')['female_parents'].keys().tolist()
    res_dict['male_virgins'] = df2.nlargest(10, 'male_virgins')['male_virgins'].keys().tolist()
    res_dict['female_virgins'] = df2.nlargest(10, 'female_virgins')['female_virgins'].keys().tolist()
    res = pd.DataFrame(res_dict, columns=res_dict.keys())
    res.to_csv(res_out)

    fig, ax = plt.subplots(figsize=(29, 6))
    sns.heatmap(df2.T)
    ax.format_coord = lambda x, y: 'x={:d}, y={:d}, z={:2f}'.format(int(np.floor(x)), int(np.floor(y), ), df.iloc[int(np.floor(y)), int(np.floor(x))])
    data_plot_utils.save_plots(plt, f'{plots_folder}/ranksum2_present_results')  # TODO
    plt.show()


    utils.write_log(f'finished ranksum_present_results')



def size_of_clusters(path_in, plots_folder='./plots_folder1/part4'):
    utils.write_log(f'start size_of_clusters')
    df = pd.read_csv(path_in, index_col=0, header=0)
    df = df.T
    df = df[df['linkage_labels'] != -1]
    df['linkage_labels'] = df['linkage_labels'].astype(float)
    df = df.astype({'linkage_labels': 'int32', 'female': 'int32', 'parent': 'int32'})
    df['group'] = df['female']*2+df['parent']
    plt.figure(figsize=(20, 10))  # width:20, height:3
    ax = pd.crosstab(df['linkage_labels'], df['group'], ).plot.bar(width=0.8, figsize=(20, 10))
    ax.tick_params(axis='x', which='both', labelsize=18)
    plt.legend(['male_naive', 'male_parent', 'female_naive', 'female_parent'])
    data_plot_utils.save_plots(plt, f'{plots_folder}/clusters_sizes')
    plt.show()
    utils.write_log(f'finished size_of_clusters')


if __name__ == '__main__':
    # split_clusters_and_groups(path_in_data='./gaba_merged_data10/gaba_stacked_2_v2.csv',
    #                           path_in_groups_data='./gaba_clustered_data11/gaba_all_clustter_stats.csv',
    #                           folder_out='gaba_groups12')

    # compere_female_vs_male_for_each_cluster(folder_in='gaba_groups12')

    # ranksum_plot(folder_in='gaba_groups12', res_out='./gaba_clustered_data11/ranksum_res.csv')
    ranksum_present_results('./gaba_clustered_data11/ranksum_res.csv', './gaba_clustered_data11/ranksum_top_10.csv')

    # size_of_clusters(path_in='./gaba_clustered_data11/gaba_all_samples_stats.csv')

