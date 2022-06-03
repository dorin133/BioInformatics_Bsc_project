import pandas as pd
import numpy as np
import seaborn as sns
import os
import datetime
import matplotlib.pyplot as plt
from itertools import islice
from sklearn import decomposition
import data_plot_utils
import utils
import time


def features_to_csv(folder_path='./raw_data', out_folder_path='./csv_data2'):
    raw_files = os.listdir(folder_path)  # list all raw files
    file_name = list(filter(lambda x: '_features.tsv' in x, raw_files))[0]
    lst = []
    input_file_path = folder_path + "/" + file_name
    f_input = open(input_file_path, 'r')
    output_file_path = out_folder_path + '/features.csv'
    for line in f_input:
        lst_to_append = []
        lst_to_append.append(line.split("\t")[0])
        lst_to_append.append(line.split("\t")[1])
        lst.append(lst_to_append)
    f_input.close()
    arr = np.array(lst)
    df = pd.DataFrame(arr, columns=['geneID', 'geneName'])
    df.index = df.index + 1 
    df.to_csv(output_file_path)

def raw_mtx_to_csv(file_path, path_out):
    print(f'status: start processing {file_path}')
    f_input = open(file_path, 'r')
    f_input.readline()
    f_input.readline()
    line = f_input.readline()
    col_size = int(line.split(' ')[1])
    table = np.zeros((27999, col_size + 1), dtype=int)
    for index, line in enumerate(f_input):
        if len(line) > 3:  # only read valid lines
            # print(line)
            curr_feature, curr_barcode, curr_value = line[:-1].split(' ')  # assuming already know the format
            curr_feature, curr_barcode, curr_value = int(curr_feature), int(curr_barcode), int(curr_value)
            table[curr_feature, curr_barcode] = curr_value

            if index % 1000000 == 0:
                print(f'status: reached line #{index}')

    f_input.close()
    print(f'status: finish processing mtx file. creating csv file')
    df = pd.DataFrame(table)
    df.drop([0], inplace=True, axis=0)
    df.drop([0], inplace=True, axis=1)
    df.to_csv(path_out)
    print(f'status: created the file "{path_out}"')
    

def prepare_metadata_single_files(folder_path='./raw_data', out_folder_path='./csv_data2'):
    raw_files = os.listdir(folder_path)  # list all raw files
    # print(raw_files)
    raw_files = list(filter(lambda x: '_barcodes.tsv' in x, raw_files))  # filter files which are not barcodes files
    # print(raw_files)
    dataset = pd.read_excel(folder_path+'/MEA_dimorphism_samples.xlsx', names=['sample_id', 'female', 'parent'], index_col=None)
    for file_name in raw_files:
        lst = []
        tmp = file_name.index('_barcodes')
        file_id = file_name[tmp-4:tmp]
        input_file_path = folder_path + "/" + file_name
        f_input = open(input_file_path, 'r')
        output_file_path = out_folder_path + "/" + file_id +'_metadata' + '.csv'
        # f_output = open(output_file_path, 'a+')
        bool_female = str(dataset[dataset['sample_id']==file_id].female.unique()[0])
        bool_parent = str(dataset[dataset['sample_id']==file_id].parent.unique()[0])
        for line in f_input:
            lst_to_append = []
            lst_to_append.append(line[:-3] + '_' + file_id)
            lst_to_append.append(file_id)
            lst_to_append.append(bool_female)
            lst_to_append.append(bool_parent)
            lst.append(lst_to_append)
        f_input.close()
        arr = np.array(lst)
        df = pd.DataFrame(arr, columns=['barcode', 'sampleID', 'female', 'parent'])
        df.index = df.index + 1 
        df.to_csv(output_file_path)


def filter_cols(path_in_file, path_out_file, min_sum_for_col=3000, min_diff_for_col=2500):
    print(f'status: start filtering {path_in_file} by col sum < {min_sum_for_col} and col number of different gens < '
          f'{min_diff_for_col}')
    df = pd.read_csv(path_in_file, index_col=0, header=0, dtype=np.int32)
    num_col_start = df.shape[1]
    df = df.loc[:, (df.sum(numeric_only=True) >= min_sum_for_col)]  # filter cols with sum less than 3000
    df = df.loc[:, ((df != 0).sum() > min_diff_for_col)]  # filter cols with less than 2500 different genes
    df.to_csv(path_out_file, sep=',')
    num_col_end = df.shape[1]
    msg = f'Note: started with {num_col_start} cols, after filtering left with {num_col_end} (filtered ' \
          f'{num_col_start-num_col_end} cols)'
    utils.write_log("filter_by_min_sum: " + path_in_file + ": " + msg)
    print(f'status: finish filtering {path_in_file}. result saved to {path_out_file}')

def filter_metadata_rows(folder_mtx_path, folder_to_metadata, out_folder_path):
    raw_files = os.listdir(folder_mtx_path)  # list all raw files
    raw_files = list(filter(lambda x: '_matrix_filtered.csv' in x, raw_files))  
    for file_name in raw_files:
        input_file_path = folder_mtx_path + "/" + file_name
        df = pd.read_csv(input_file_path, index_col=0, header=0, dtype=np.int32)
        tmp = file_name.index('_matrix_filtered')
        file_id = file_name[tmp-4:tmp]

        path_to_metadata = folder_to_metadata + '/' + file_id + '_metadata.csv'
        path_output = out_folder_path + '/' + file_id + '_metadata.csv'
        print(f'status: start filtering {path_to_metadata}')

        df_metadata = pd.read_csv(path_to_metadata, index_col=0, header=0)
        results = map(int, df.columns.tolist())
        df_metadata = df_metadata.loc[results]
        df_metadata.to_csv(path_output[:-4]+'_filtered.csv', sep=',')


def filter_common_and_rare_gens(path_stacked_mtx_file='./merged_data5/stacked_normalized_mtx.csv', path_out_file=
'./merged_data/stacked_normalized_filtered_mtx.csv'):
    df = pd.read_csv(path_stacked_mtx_file, index_col=0, header=0)
    hist_row_non_zeros = (df != 0).sum(axis=1)
    df_filtered = df[5 < hist_row_non_zeros]
    hist_row_non_zeros = (df_filtered != 0).sum(axis=1)
    df_filtered = df_filtered[hist_row_non_zeros < df.shape[0] / 2]
    print(f'status: filter_common_and_rare_gens: filtered {df.shape[0]-df_filtered.shape[0]} genes. filtered csv saved '
          f'as {path_out_file}')
    df_filtered.to_csv(path_out_file, sep=',')


def normalize_data(path_in_file, path_out_file, alpha=20000):
    df = pd.read_csv(path_in_file, index_col=0, header=0)
    df = np.ceil(alpha*(df/df.sum()))
    # df = np.ceil(alpha*df/np.linalg.norm(df, axis=0))
    # print("problem there's a problem with the normalization function!!!")
    path_out_file = path_out_file[:-13]+'_normalized.csv'
    print(f'status: finish normalizing {path_in_file}. result saved to {path_out_file}')
    df.to_csv(path_out_file, sep=',')
    # don't forget to write critical info to log


def calc_and_plot_cv(path_stacked_mtx_file, path_to_features_csv, path_out, plots_folder='./plots_folder1'):
    df = pd.read_csv(path_stacked_mtx_file, index_col=0, header=0)
    print(f'status: start calc_and_plot_cv. df original shape is {df.shape}')

    # calculating cv and mean for each gene
    cv_res = df.apply(lambda x: np.std(x, ddof=1) / np.mean(x), axis=1)
    mean_res = df.apply(lambda x: np.mean(x), axis=1)
    # cv_res_pd = cv_res_pd.dropna()
    # cv_res = cv_res_pd
    # mean_res = mean_res[mean_res > 0]
    # msg = "cv and mean shape after removing zeros: " + str(cv_res.shape)+"\n"
    # print(msg)
    # cv_res = cv_res.to_numpy(dtype=np.float32, copy=True)
    # mean_res = mean_res.to_numpy(dtype=np.float32, copy=True)

    # apply log to the mean and cv
    cv_res = np.log2(cv_res)
    mean_res = np.log2(mean_res)

    # plot the scatter and the best linear line (named p) to fit it
    plt.scatter(mean_res, cv_res, c='green', s=0.4, marker="o")
    p = np.poly1d(np.polyfit(mean_res, cv_res, 1))
    print(p)
    # should be around [-0.5, 0.5]
    # now it is: [-0.5, 0.75]
    plt.plot(np.unique(mean_res), p(np.unique(mean_res)))

    # find the 100 farthest genes from p
    dist_cv = (cv_res - p(mean_res))
    dist_cv = (cv_res - p(mean_res))
    # dist_cv.index contains the real indeces of the genes (by the features table)
    dist_cv_dict = dict(zip(dist_cv.index, dist_cv.values))
    # the next line returns the real indeces (by the features table) of the 100 genes with the highest cv
    dist_cv_dict_100 = sorted(dist_cv_dict, key=dist_cv_dict.get, reverse=True)[:100]
    df_features = pd.read_csv(path_to_features_csv, index_col=0, header=0)

    # dist_idx = np.argsort(dist_cv)[-100:]
    # real_idx = []
    # TODO: make the new indeces the index of this stacked table!!!
    # print("dist_idx", dist_idx)
    # print("cv_res_pd.index:", cv_res_pd.index)
    # for index, j in enumerate(df.index):
    #     if index in dist_idx:
    #         # print("dist_idx: "+ str(index)+" matches the real idx: "+ str(j-2))
    #         real_idx.append(j-2)
    # print("real_index:", real_idx)
    labels = df_features.loc[dist_cv_dict_100].geneName.unique()
    i = 0
    print("the 100 farthest genes in the cv plot are: ")
    print(labels)
    # add to the plot the names of the farthest genes
    for x, y in zip(mean_res.loc[dist_cv_dict_100], cv_res.loc[dist_cv_dict_100]):
        plt.annotate(labels[i],  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 2),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
        i += 1
    plt.title("log(mean) as function of log(cv) for each gene")
    plt.xlabel("log(mean)")
    plt.ylabel("log(cv)")
    data_plot_utils.save_plots(plt, f'{plots_folder}/cv_plot')
    plt.show()

    msg = f"calc_and_plot_cv: finished plotting mean as a function of cv for each gene\n The 100 genes we cleaned " \
          f"are {labels}"
    utils.write_log(msg)

    # find knee for threshold filter
    sorted_dict_cv = {k: v for (k, v) in dist_cv.items() if v >= 0}
    sorted_dict_cv = dict(sorted(sorted_dict_cv.items(), key=lambda item: item[1], reverse=True))

    # sorted_dict_cv = np.flip(np.sort(dist_cv[dist_cv>0]))
    y_values = (list(sorted_dict_cv.values()) - np.amin(list(sorted_dict_cv.values()))) / np.amax(
        list(sorted_dict_cv.values()))
    x_values = np.arange(1 / len(sorted_dict_cv), 1 + 1 / len(sorted_dict_cv), 1 / len(sorted_dict_cv))
    plt.plot(x_values, y_values)
    knee_val = 100
    knee_point = None
    genes_threshold = -1
    for i, point in enumerate(zip(x_values, y_values)):
        tmp_dist = np.sqrt(point[0] ** 2 + point[1] ** 2)
        if tmp_dist < knee_val:
            knee_val = tmp_dist
            knee_point = point
            genes_threshold = i
    print(f'Found knee point (closest to the origin) at {knee_point}')
    print(f'genes_threshold is {genes_threshold}')
    plt.annotate("knee",  # this is the text
                 knee_point,  # these are the coordinates to position the label
                 textcoords="offset points",  # how to position the text
                 xytext=(0, 0),  # distance from text to points (x,y)
                 ha='center')  # horizontal alignment can be left, right or center

    plt.axhline(y=knee_point[1], color='r', linestyle='-')
    plt.title(f'CV distance (absolute) density. recommend threshold={round(knee_point[1], 4)}')
    data_plot_utils.save_plots(plt, f'{plots_folder}/cv_knee_threshold')
    plt.show()

    # df_threshold = df[dist_cv_absolute <= knee_point[1]]  # TODO double check this
    genes_survived = dict(islice(sorted_dict_cv.items(), genes_threshold))
    df_threshold = df.loc[genes_survived.keys()]  # TODO double check this

    # labels_all = df_features.loc[genes_survived.keys()].geneName.unique()
    # print("the genes who survived are: ")
    # print(labels_all)
    # list_for_sanity_check = ['Snap25','Stmn2','Gad2','Slc32a1','Slc17a7','Slc17a6','Sst','Sim1','Tac2','Ttr','Foxj1','Acta2','Flt1','Cldn5','Aqp4','Plp1']
    # print(list_for_sanity_check - labels_all)
    # print(len(labels_all - labels_all))
    print(f'Status: removing rows (gens) which their distance from their CV point to the fit-line is greater then the '
          f'threshold={round(knee_point[1], 4)}')

    manually_remove = ['Xist', 'Tsix', 'Eif2s3y', 'Ddx3y', 'Uty', 'Kdm5d']
    print(f'Also manually removing {manually_remove}')
    drop_idx = []
    for index, row in df_features.iterrows():  # index start from 1
        if row['geneName'] in manually_remove and index in df_threshold.index:
            drop_idx.append(index)
    df_threshold = df_threshold.drop(drop_idx)

    df_threshold.to_csv(path_out, sep=',')
    print(f'That removed {df.shape[0] - df_threshold.shape[0]} genes. The new csv file saved as {path_out}')
    print(f'We were left with {df_threshold.shape[0]} genes.')
    print(f'status: finish calc_and_plot_cv. the new df shape is {df_threshold.shape}')
    # print(f' which are: {df_features.loc[genes_survived.keys()].geneName.unique()}')


def normalize_list_numpy(list_np, scale=1):
  min_t = np.min(list_np)
  max_t = np.max(list_np)
  return scale * (list_np - min_t) / (max_t - min_t)


def pca_norm_knee(path_in, path_out, plots_folder='./plots_folder1'):
    df = pd.read_csv(path_in, index_col=0, header=0)
    utils.write_log(f"pca_norm_knee: starting. original data shape is {df.shape} (we Transpose this in a moment)")

    # TODO holy bug or coesintance ??
    df_t = df.T  # TODO this way PCA is 14. according to the last call with Amit this is the right way
    # print("!", df_t.shape)
    # print("!!", df_t.mean(), "\n%%", df_t.mean().shape)
    # print("!!", df_t.mean(axis=0), "\n%%", df_t.mean(axis=0).shape)
    # print("!!", df_t.std(axis=0), "\n%%", df_t.std(axis=0).shape)
    df_t = np.log2(df_t+1)
    df_t = ((df_t - df_t.mean(axis=0)) / df_t.std(axis=0))  # TODO verify this

    # df = np.log2(df + 1)  # TODO we are not sure if the log and norm should be on rows or on columns. this way PCA is 18
    # df = (df - df.mean() / df.std())
    # df_t = df.T

    curr_n = df_t.shape[1]  # all genes
    pca = decomposition.PCA(n_components=curr_n)
    _ = pca.fit_transform(df_t)
    explain = pca.explained_variance_

    explain = normalize_list_numpy(explain)
    explainx_axe_norm = normalize_list_numpy([t for t in range(len(explain))])
    knee_val = 1000
    knee_point = 0
    knee_x = 0
    number_of_values_bigger_than_knee = 0
    print("explain:\n", explain)
    for index, (x, y_val) in enumerate(zip(explainx_axe_norm, explain)):
        tmp_dist = np.sqrt(x ** 2 + y_val ** 2)
        if tmp_dist < knee_val:
            knee_val = tmp_dist
            number_of_values_bigger_than_knee = index+1
            knee_point = y_val
            knee_x = x
    print('knee_point:', knee_point, f". coordinates: ({knee_x}, {knee_point})")
    plt.plot(explainx_axe_norm, explain)
    plt.title(f'PCA Explained Variance. knee={round(knee_point, 4)}\nonly {number_of_values_bigger_than_knee} '
              f'values are bigger than the knee value')
    plt.ylabel('Explained Variance')
    plt.xlabel('Components')
    plt.axhline(y=knee_point, color='r', linestyle='-')
    data_plot_utils.save_plots(plt, f'{plots_folder}/pca_explain')
    plt.show()

    # now take only the top PCA features
    pca = decomposition.PCA(n_components=number_of_values_bigger_than_knee)
    principal_components = pca.fit_transform(df_t)
    new_cols = [f'pca_feature_{i}' for i in range(1, number_of_values_bigger_than_knee + 1)]
    principal_df = pd.DataFrame(data=principal_components, columns=new_cols, index=df_t.index)
    print(f'Now PCA with {number_of_values_bigger_than_knee}. explain now is:\n', pca.explained_variance_)
    print(f"notice those are just the top {number_of_values_bigger_than_knee} for the prev PCA explain we plot")
    principal_df.T.to_csv(path_out, sep=',')
    utils.write_log(f'finish PCA. left with {number_of_values_bigger_than_knee} values. current data shape is '
                    f'{principal_df.shape} (Transposed). saved to {path_out}')
