import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import data_plot_utils
import time
import utils


def show_intresting(path_in_stack, path_in_dbscan, path_to_features_csv, gen_list):
    df_stack = pd.read_csv(path_in_stack, index_col=0, header=0)
    # print(df_stack.isna().sum(axis=1))  # TODO not sure why is that
    # df_stack.fillna(0, inplace=True)
    # print(df_stack.isna().sum(axis=1))

    df_dbscan = pd.read_csv(path_in_dbscan, index_col=0, header=0)
    df = pd.concat([df_stack, df_dbscan])
    df = df.T

    features = pd.read_csv(path_to_features_csv, header=0)
    features.set_axis(['num', 'geneID', 'geneName'], axis=1, inplace=True)
    features.set_index('geneName', inplace=True)
    for gen_name in gen_list:

        gen_id = int(features.at[gen_name, 'num'])
        if gen_id not in df.columns:
            print(f"!!! gen {gen_name} (gen id {gen_id}) is not in the stacked matrix data")
            continue
        print(f'gen id for {gen_name} is {gen_id}')
        df['is_intresting'] = df.apply(lambda x: x[gen_id] != 0, axis=1)

        plt.figure(figsize=(9, 9))
        sns.scatterplot(
            x="tsne-2d-one",
            y="tsne-2d-two",
            hue="is_intresting",
            # palette = color,
            data=df,
            legend="auto",
            alpha=0.3
        )

        # for point in aimed:
        #   plt.plot(point[0], point[1], 's')
        #   plt.text(point[0], point[1], point[2], horizontalalignment='left', size='medium', color='black', weight='semibold')

        plt.title(f"Scatter for gen {gen_name} (gen id {gen_id}). Found {df['is_intresting'].sum()} sampels with this gen")
        plt.show()



def cv_v2(path_stacked_mtx_file, path_to_features_csv, path_out, plots_folder='./plots_folder1'):
    utils.write_log(f'start cv_v2')
    time_start = time.time()
    df = pd.read_csv(path_stacked_mtx_file, index_col=0, header=0)
    utils.write_log(f'df original shape is {df.shape} (just loading it took {time.time()-time_start} seconds)')

    # calculating cv and mean for each gene
    cv_res = df.apply(lambda x: np.std(x, ddof=1) / np.mean(x), axis=1)
    mean_res = df.apply(lambda x: np.mean(x), axis=1)
    # apply log to the mean and cv
    cv_res = np.log2(cv_res)
    mean_res = np.log2(mean_res)
    p = np.polyfit(mean_res, cv_res, 1)
    m, b = p[0], p[1]
    utils.write_log(f'poly 1d fit: m={m}, b={b}')

    dist_cv = (cv_res - (m * mean_res + b))
    top_n = 100
    top_k = dist_cv.nlargest(top_n)
    top_k_val = top_k.min()

    feat = pd.read_csv(path_to_features_csv, header=0)
    feat.set_axis(['num', 'geneID', 'geneName'], axis=1, inplace=True)
    feat.set_index('num', inplace=True)

    t = dist_cv[dist_cv > 0]
    t.plot.hist(bins=200)
    plt.show()
    t.plot.kde()
    plt.show()

    # ############# TODO add dyn esp
    t = dist_cv[dist_cv > 0].value_counts(normalize=True, bins=50)
    # t = dist_cv[dist_cv>0].value_counts(bins=100)
    t = t / t.max()
    knee_val = 999
    dist_threshold = -999
    for x, y in t.items():
        avg_x = (x.left + x.right) / 2
        tmp_dist = np.sqrt((avg_x ** 2) + (y ** 2))
        if tmp_dist < knee_val:
            knee_val = tmp_dist
            dist_threshold = avg_x
    utils.write_log(f'knee_point={dist_threshold}')
    plt.title(f'CV distance density. Recommend threshold={round(dist_threshold, 4)}')
    data_plot_utils.save_plots(plt, f'{plots_folder}/cv_knee_plot')
    plt.show()
    # dist_threshold = 0.02  # TODO
    # ############# TODO add dyn esp

    p = np.polyfit(mean_res, cv_res, 1)
    m, b = p[0], p[1]
    plt.scatter(mean_res, cv_res, c='turquoise', s=0.4, marker="o")
    plt_lim = (plt.xlim())
    x = np.linspace(plt_lim[0], plt_lim[1], 1000)
    plt.plot(x, m * x + b)
    plt.plot(x, m * x + b + dist_threshold, color='red')
    plt.plot(x, m * x + top_k_val + dist_threshold, color='orange')

    for gen_id in top_k.index:
        gen_name = feat.at[gen_id, 'geneName']
        plt.text(mean_res[gen_id], cv_res[gen_id], gen_name, horizontalalignment='left', size='medium', color='black',
                 weight='semibold')
    plt.title("log(mean) as function of log(cv) for each gene")
    plt.xlabel("log(mean)")
    plt.ylabel("log(cv)")
    data_plot_utils.save_plots(plt, f'{plots_folder}/cv_plot')
    plt.show()

    left = dist_cv[dist_cv < dist_threshold]
    feat['num'] = feat.index
    feat.set_index('geneName', inplace=True)

    manually_removed = ['Xist', 'Tsix', 'Eif2s3y', 'Ddx3y', 'Uty', 'Kdm5d']
    for gen_name in manually_removed:
        gen_id = feat.at[gen_name, 'num']
        if gen_id in left:
            left.drop([gen_id], inplace=True)

    # left = dist_cv[dist_cv < dist_threshold]
    left = dist_cv[dist_cv > dist_threshold]  # TODO

    feat['geneName2'] = feat.index
    feat.set_index('num', inplace=True)

    left_genes = [(gen_id, feat.at[gen_id, 'geneName2']) for gen_id in left.index]
    utils.write_log(f'left with the following {len(left_genes)} gens (gen_id, gen_name): {left_genes}')

    df2 = df.filter(items=left.index, axis=0)

    utils.write_log(f'cv summary: That removed {df.shape[0] - df2.shape[0]} genes')
    df2.to_csv(path_out, sep=',')
    utils.write_log(f'status: finish cv_v2. the new df shape is {df2.shape}, saved to {path_out}. running time for this'
                    f' function was {time.time()-time_start} seconds')


# ml_processing.show_intresting(path_in_stack='./merged_data5/stacked_3_v3.csv',
#                               path_in_dbscan='./merged_data5/dbscan.csv',
#                               path_to_features_csv='./csv_data2/features.csv', gen_list= ['Snap25','Stmn2','Gad2'
#         ,'Slc32a1', 'Slc17a7','Slc17a6','Sst','Sim1','Tac2', 'Ttr','Foxj1','Acta2','Flt1','Cldn5', 'Aqp4','Plp1'])