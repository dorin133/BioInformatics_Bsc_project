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

def barplot_dataset_size(plots_folder= './plots_folder1/report_missing_plots'):
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
    # cell_to_label = pd.read_csv(path_in, index_col=0, header=0).T
    hist_group = {0: 809, 1: 1057, 2: 910, 3: 865, 4: 612, 5: 855, 6: 909, 7: 587, 8: 914, 9: 957, 10: 519, 11: 558, 12: 659, 13: 634, 14: 680, 15: 654}
    hist_df = pd.DataFrame.from_dict(hist_group, orient='index')
    hist_df = hist_df.T
    hist_df.columns = ['35_1', '35_2', '36_1', '36_2', '37_1', '37_2', '38_1', '38_2', '51_1', '51_2', '51_3', '51_4', '52_1', '52_2', '52_3', '52_4']
    # hist_df.sort_index(axis=0, inplace=True)
    hist_df = hist_df.T
    ax = hist_df.plot.bar(stacked=True, figsize=(16, 10), rot=0)
    plt.title('GABAergic Dataset sample size per mice', fontdict={'fontsize': 25}, pad=30)
    ax.tick_params(axis='x', which='both', labelsize=18)
    ax.tick_params(axis='y', which='both', labelsize=18)
    ax.legend(["amount"], fontsize=20)
    data_plot_utils.save_plots(plt, f'{plots_folder}/clusters_bar_groups')
    plt.show()
    pass

def ranksum_present_results(path_in, res_out, plots_folder='./plots_folder1/report_missing_plots'):
    # utils.write_log(f'start ranksum_present_results')
    df = pd.read_csv(path_in)
    df = df[df['Unnamed: 0'].str.len() > 10]
    df.set_index('Unnamed: 0', inplace=True)

    # fig, ax = plt.subplots(figsize=(35, 70))
    # sns.heatmap(df, cbar=False)
    # # ax.format_coord = lambda x, y: 'x={:d}, y={:d}, z={}'.format(int(np.floor(x)), int(np.floor(y), ), f'{int(np.floor(x))}_{int(np.floor(y))}')
    #
    # data_plot_utils.save_plots(plt, f'{plots_folder}/ranksum1_present_results')  # TODO
    # plt.show()

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

def cdf_plot_gender(plots_folder='./plots_folder1/report_missing_plots'):
    x_values = np.arange(0, 46, 1)
    y_values = [0.22742788682353887, 1.0, 0.001307087477083635 , 0.45703430056411787, 0.31563770589905393 , 0.47914575890053945 ,
                1.0, 0.011432667964698218 , 0.8742048669763107 , 0.13553870494625975 , 0.29497297695966207, 1.0 ,
                0.038525423611718956, 0.00196326733561214 , 1.0, 0.9115790120181684, 0.0, 0.023551508022681134 , 0.2106130873844756 , 1.0,
                0.9652868684098398, 0.00035943145671968324 , 0.0884885466729105 , 0.3705363457901838 , 0.13661319546885475 , 0.7056453227878836 ,
                1.0, 0.048716536050546155 , 0.0015852884916783827 , 0.21703727705991505 , 0.000311695102655718 , 0.7783152754678014 , 0.0038521375528294266 ,
                0.992263191288008, 0.42726034853305983 , 0.3915040724066826 , 0.0, 0.08304834520090798 , 0.3509228487681816 , 0.003977040855760894 ,
                0.019363340425667896, 0.12234203342039485 , 0.6176749626480671 , 0.24135849951161303 , 0.33057863091256257 , 0.000282970408967409 ]
    plt.bar(x_values, y_values)
    plt.ylabel('(1 - p) of hypergeometric CDF')
    plt.xlabel('Cluster Index By Linkage')
    # plt.title("GABAergic Clusters' Group Enrichment Female / Male")
    # data_plot_utils.save_plots(plt, f'{plots_folder}/clusters_cdf_gender')
    plt.show()

def cdf_plot_parenting(plots_folder='./plots_folder1/report_missing_plots'):
    x_values = np.arange(0, 46, 1)
    y_values = [0.4406839148433357, 0.0, 0.7536658492987034 , 0.3024413722134365, 0.4517670083756584 , 0.11148789396878178 ,
                0.004399699056344408, 0.9996108978070214, 0.05946748514050715 , 0.9153411926459802 , 0.5839844334227615, 1.0 ,
                0.14887247229523137, 0.0 , 0.018332667344235776, 0.9954177051268971, 0.0018632890473329056, 0.9510029733533885 ,  0.20989615614648982 ,0.03978149247316465 ,
                0.8731062604341768, 0.8409648364871036 , 0.0817334696177966 , 0.991233357734625 , 0.9815838294082755 , 0.0012819714996918918 ,
                0.9764382319298536, 0.9850665978869948 , 0.9998590598124305 , 0.9899020426382378 , 0.7215715248560405 , 0.04823888116573416 , 0.9976966466043441 ,
                0.0011053261684552673, 0.9801523064539261 , 0.998470164122181 , 0.9999999961898879 , 0.9470555077513186 , 0.0 , 0.5268852067244505 ,
                0.3200329668098534, 0.5939402672952351, 0.011105211349708655 , 0.9113246499963139 , 0.43213742988486126 , 1.0 ]
    plt.bar(x_values, y_values)
    plt.ylabel('(1 - p) of hypergeometric CDF')
    plt.xlabel('Cluster Index By Linkage')
    # plt.title("GABAergic Clusters' Group Enrichment Parent / Naive")
    # data_plot_utils.save_plots(plt, f'{plots_folder}/clusters_cdf_parenting')
    plt.show()



if __name__ == '__main__':
    cdf_plot_gender()
    cdf_plot_parenting()
    print("Done")