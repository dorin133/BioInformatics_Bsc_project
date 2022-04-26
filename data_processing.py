import pandas as pd
import numpy as np
import seaborn as sns
import os
import datetime
import matplotlib.pyplot as plt

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
    df = df.loc[:, ((df != 0).sum() > min_diff_for_col)]  # filter cols with less than 2500 different gens
    df.to_csv(path_out_file, sep=',')
    num_col_end = df.shape[1]
    msg = f'Note: started with {num_col_start} cols, after filtering left with {num_col_end} (filtered ' \
          f'{num_col_start-num_col_end} cols)'
    print(msg)
    f = open(f'./ml_run_logs.txt', 'a+')
    msg = str(datetime.datetime.now()) + " filter_by_min_sum: " + path_in_file + ": " + msg + "\n"
    f.write(msg)
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
    df = np.ceil(alpha*df/np.linalg.norm(df, axis=0))
    path_out_file = path_out_file[:-13]+'_normalized.csv'
    print(f'status: finish normalizing {path_in_file}. result saved to {path_out_file}')
    df.to_csv(path_out_file, sep=',')
    # don't forget to write critical info to log


def calc_and_plot_cv(path_stacked_mtx_file='./merged_data5/stacked_normalized_filtered_mtx.csv', path_to_features_csv=
'./csv_data2/features.csv', path_out='./merged_data5/stacked_normalized_filtered_threshold_mtx.csv', plots_folder='./plots_folder1'):
    df = pd.read_csv(path_stacked_mtx_file, index_col=0, header=0)
    
    # calculating cv and mean for each gene
    cv_res_pd = df.apply(lambda x: np.std(x, ddof=1) / np.mean(x), axis=1)
    mean_res = df.apply(lambda x: np.mean(x), axis=1) 
    cv_res = cv_res_pd.dropna()
    mean_res = mean_res[mean_res > 0]
    # msg = "cv and mean shape after removing zeros: " + str(cv_res.shape)+"\n"
    # print(msg)
    cv_res = cv_res.to_numpy(dtype=np.float32, copy=True)
    mean_res = mean_res.to_numpy(dtype=np.float32, copy=True)

    # apply log to the mean and cv
    cv_res = np.log10(cv_res)
    mean_res = np.log10(mean_res)

    # plot the scatter and the best linear line (named p) to fit it
    plt.scatter(mean_res, cv_res, c='green', s=0.4, marker="o")
    p = np.poly1d(np.polyfit(mean_res, cv_res, 1))
    plt.plot(np.unique(mean_res), p(np.unique(mean_res)))

    # find the 100 farthest genes from p
    dist_cv = cv_res - p(mean_res)
    dist_idx = np.argsort(dist_cv)[-100:]
    df_features = pd.read_csv(path_to_features_csv, index_col=0, header=0)
    labels = df_features.loc[dist_idx+1].geneName.unique()  # notice the "+1" to fix the diff between the two
    i = 0
    # add to the plot the names of the farthest genes
    for x, y in zip(mean_res[dist_idx], cv_res[dist_idx]):
        plt.annotate(labels[i],  # this is the text
                    (x, y),  # these are the coordinates to position the label
                    textcoords="offset points",  # how to position the text
                    xytext=(0, 2),  # distance from text to points (x,y)
                    ha='center')  # horizontal alignment can be left, right or center
        i += 1
    plt.title("log(mean) as function of log(cv) for each gene")
    plt.xlabel("log(mean)")
    plt.ylabel("log(cv)")
    plt.savefig(f'{plots_folder}/cv_plot{str(datetime.datetime.now().time())[:8].replace(":", "_")}.png')
    plt.show()

    f = open(f'./ml_run_logs.txt', 'a+')
    msg = str(datetime.datetime.now()) + f" calc_and_plot_cv: finished plotting mean as a function of cv for each gene" \
                                         f"\n The 100 genes we cleaned are {labels}\n"
    f.write(msg)
    f.close()

    # find knee for threshold filter
    # dist_cv_absolute = np.absolute(dist_cv)  # TODO option 2
    # ax = sns.distplot(pd.DataFrame(dist_cv_absolute), hist=True)
    ax = sns.distplot(pd.DataFrame(dist_cv), hist=True)
    # ax = sns.distplot(pd.DataFrame(dist_cv[dist_cv>0]), hist=True)
    line = ax.lines[0]
    knee_val = 100
    knee_point = None
    for point in line.get_xydata():
        tmp_dist = point[0]**2 + point[1]**2
        if point[0] > 0.01 and tmp_dist < knee_val:  # TODO the 'point[0] > 0.01' is not the best solution...
            knee_val = tmp_dist
            knee_point = point
    print(f'Found knee point (closest to the origin) at {knee_point}')
    plt.annotate("knee",  # this is the text
                 knee_point,  # these are the coordinates to position the label
                 textcoords="offset points",  # how to position the text
                 xytext=(0, 0),  # distance from text to points (x,y)
                 ha='center')  # horizontal alignment can be left, right or center


    plt.axhline(y=knee_point[1], color='r', linestyle='-')
    plt.title(f'CV distance (absolute) density. recommend threshold={round(knee_point[1], 4)}')
    plt.savefig(f'{plots_folder}/cv_knee_threshold{str(datetime.datetime.now().time())[:8].replace(":", "_")}.png')
    plt.show()

    # df_threshold = df[dist_cv_absolute <= knee_point[1]]  # TODO double check this
    df_threshold = df[dist_cv <= knee_point[1]]  # TODO double check this
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
    print(f'That removes {df.shape[0]-df_threshold.shape[0]} genes. The new csv file saved as {path_out}')


