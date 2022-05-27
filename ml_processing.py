import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import data_plot_utils
import time
import utils


def tSNE(path_in, path_to_MEA = './raw_data/MEA_dimorphism_samples.xlsx', path_out='./merged_data5/tsne.csv',
         plots_folder='./plots_folder1'):
    utils.write_log('start tSNE')
    df_f_m_index = pd.read_excel(path_to_MEA)
    f_list, m_list = [], []
    p_list, no_p_list = [], []
    for index, row in df_f_m_index.iterrows():
        if row['female'] == 1:
            f_list.append(row.iloc[0])
        else:
            m_list.append(row.iloc[0])
        if row['parent'] == 1:
            p_list.append(row.iloc[0])
        else:
            no_p_list.append(row.iloc[0])
    df = pd.read_csv(path_in, index_col=0, header=0)
    df = df.T

    original_cols = df.columns
    df['gender'] = ''
    df['parent'] = 0
    df['labels'] = ''  # femal/male and parent/no_parent
    mapping = {0: 'male_no_parent', 1: 'male_parent', 2: 'female_no_parent', 3: 'female_parent'}
    for index, row in df.iterrows():
        id = index[-4:]
        # print(id)
        tmp = 0
        if id in f_list:
            df.at[index, 'gender'] = 'female'
            tmp += 2
        else:
            df.at[index, 'gender'] = 'male'

        if id in p_list:
            df.at[index, 'parent'] = 1
            tmp += 1

        df.at[index, 'labels'] = mapping[tmp]

    pca12 = df[['pca_feature_1', 'pca_feature_2']]
    pca12 = (pca12 - pca12.mean(axis=0) / pca12.std(axis=0))
    pca12 = 10e-4 * pca12.to_numpy()

    time_start = time.time()
    tsne = TSNE(n_components=2, init=pca12, metric='correlation', verbose=1, perplexity=60, n_iter=1500,
                early_exaggeration=20)
    tsne_pca_results = tsne.fit_transform(df[original_cols])
    # print some info regarding the tSNE
    utils.write_log(f't-SNE done! Time elapsed: {time.time()-time_start} seconds')
    df['tsne-2d-one'] = tsne_pca_results[:,0]
    df['tsne-2d-two'] = tsne_pca_results[:,1]

    # plotting and saving the tSNE
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one",
        y="tsne-2d-two",
        hue="labels",
        # palette=sns.color_palette("gist_heat_r", 4),
        palette=['tab:blue', 'tab:green', 'tab:orange', 'tab:red'],
        data=df,
        legend="full",
        alpha=0.3
    )
    plt.title(f't-SNE in 2d')
    data_plot_utils.save_plots(plt, f'{plots_folder}/tSNE_2d_4colors')
    plt.show()

    # plotting and saving the tSNE
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one",
        y="tsne-2d-two",
        hue="gender",
        palette=['tab:blue', 'tab:orange'],
        data=df,
        legend="full",
        alpha=0.3
    )
    plt.title(f't-SNE in 2d')
    data_plot_utils.save_plots(plt, f'{plots_folder}/tSNE_2d_2colors')
    plt.show()

    df.T.to_csv(path_out, sep=',')
    utils.write_log(f'finish tSNE. new data with the new two tSNE cols + 3 labels cols is in shape {df.shape}. saved '
                    f'new data to {path_out}')


def tSNE_3d(path_in, path_to_MEA='./raw_data/MEA_dimorphism_samples.xlsx', path_out='./merged_data5/tsne.csv',
         plots_folder='./plots_folder1'):
    df_f_m_index = pd.read_excel(path_to_MEA)
    # print(df_f_m_index)
    f_list, m_list = [], []
    p_list, no_p_list = [], []
    for index, row in df_f_m_index.iterrows():
        if row['female'] == 1:
            f_list.append(row.iloc[0])
        else:
            m_list.append(row.iloc[0])
        if row['parent'] == 1:
            p_list.append(row.iloc[0])
        else:
            no_p_list.append(row.iloc[0])
    df = pd.read_csv(path_in, index_col=0, header=0)
    df = df.T
    df['female'] = 0
    df['parent'] = 0
    df['category_female_parent'] = 0  # female/male += 2, parent/no_parent += 1
    for index, row in df.iterrows():
        id = index[-4:]
        if id in f_list:
            df.at[index, 'female'] = 1
            df.at[index, 'category_female_parent'] += 2
        if id in p_list:
            df.at[index, 'parent'] = 1
            df.at[index, 'category_female_parent'] += 1
    df_subset3 = df.copy()
    time_start = time.time()
    tsne_3d = TSNE(n_components=3, verbose=0, perplexity=40, n_iter=300)
    tsne_pca_results3 = tsne_3d.fit_transform(df_subset3)
    # print some info regarding the tSNE in 3d
    utils.write_log(f't-SNE in 3d done! Time elapsed: {time.time()-time_start} seconds')

    df_subset3['tsne-3d-one'] = tsne_pca_results3[:,0]
    df_subset3['tsne-3d-two'] = tsne_pca_results3[:,1]
    df_subset3['tsne-3d-three'] = tsne_pca_results3[:,2]

    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    ax.scatter(
        xs=df_subset3["tsne-3d-one"],
        ys=df_subset3["tsne-3d-two"],
        zs=df_subset3["tsne-3d-three"],
        c=df_subset3["category_female_parent"],
        cmap='Spectral',
    )
    ax.view_init(elev=20, azim=150)
    ax.set_xlabel('tsne-3d-one')
    ax.set_ylabel('tsne-3d-two')
    ax.set_zlabel('tsne-3d-three')
    plt.title(f't-SNE in 3d')
    data_plot_utils.save_plots(plt, f'{plots_folder}/tSNE_3d')
    plt.show()
