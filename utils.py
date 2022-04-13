import os
import pandas as pd
import numpy as np
import datetime


def do_not_change_name(name):
    return name


def do_not_filter_files(files_path):
    return files_path


def run_func_for_all(func_to_run, folder_in_path, folder_out_path, which_files=do_not_filter_files,
                     rename_out_files=do_not_change_name):
    raw_files = os.listdir(folder_in_path)  # list all raw files
    chosen_files = list(filter(which_files, raw_files))
    chosen_files.sort()
    for file_name in chosen_files:
        new_file_name = rename_out_files(file_name)
        path_out_file_with_name = folder_out_path + '/' + new_file_name
        func_to_run(folder_in_path + "/" + file_name, path_out_file_with_name)


def stack_csv_together(folder_path, out_file_path='./merged_data/stacked_mtx.csv'):
    chosen_files = os.listdir(folder_path)  # list all raw files
    # print(raw_files)
    # raw_files = list(filter(lambda x: '_matrix2.csv' in x, raw_files))
    chosen_files = list(filter(lambda x: 'matrix_normalized.csv' in x, chosen_files))
    chosen_files.sort()
    print('status: stack_csv_together for', chosen_files)

    stacked_csv = pd.read_csv(folder_path + "/" + chosen_files[0], index_col=0, header=0)
    log_info = []
    sum_index = stacked_csv.shape[1] + 1
    log_info.append((chosen_files[0], 1))
    for file in chosen_files[1:]:
        tmp = pd.read_csv(folder_path + "/" + file, index_col=0, header=0)
        stacked_csv = pd.concat([stacked_csv, tmp], axis=1)

        log_info.append((file, sum_index))
        sum_index += tmp.shape[1]

    print(log_info)
    f = open(f'./ml_run_logs.txt', 'a+')
    msg = str(datetime.datetime.now()) + " stack_csv_together: " + log_info.__str__() + "\n"
    f.write(msg)

    stacked_csv.to_csv(out_file_path)
    print('status: finish stack_csv_together. the new concat file called', out_file_path)

def merge_all_metadata(folder_path='./filtered_data', out_folder_path='./merged_data'):
    chosen_files = os.listdir(folder_path)  # list all raw files
    chosen_files = list(filter(lambda x: '_metadata_filtered.csv' in x, chosen_files)) 
    chosen_files.sort()
    print('status: merge_all_metadata for', chosen_files)
    output_file_path = out_folder_path + "/all_samples_metadata.csv"
    metadatas_csv = pd.read_csv(folder_path + "/" + chosen_files[0], index_col=0, header=0)
    for file in chosen_files[1:]:
        tmp = pd.read_csv(folder_path + "/" + file, index_col=0, header=0)
        metadatas_csv = pd.concat([metadatas_csv, tmp])
    metadatas_csv.to_csv(output_file_path)
    print('status: finish merge_all_metadata. the new concat file called', output_file_path)


def find_indeces_of_gene(folder_path='./raw_csv_data', gene_to_filter='mt-'):
    raw_files = os.listdir(folder_path)  # list all raw files
    chosen_files = list(filter(lambda x: 'features.csv' in x, raw_files))
    features_csv = pd.read_csv(folder_path + "/" + chosen_files[0], index_col=0, header=0)
    return np.array(features_csv[features_csv['geneName'].str.startswith(gene_to_filter)].index)


