import os
import time

import pandas as pd
import numpy as np
import datetime

import matplotlib.pyplot as plt

def do_not_change_name(name):
    return name


def do_not_filter_files(files_path):
    return files_path


def run_func_for_all(func_to_run, folder_in_path, folder_out_path, which_files=do_not_filter_files,
                     rename_out_files=do_not_change_name):
    write_log(f'start run_func_for_all with {func_to_run.__name__}')
    raw_files = os.listdir(folder_in_path)  # list all raw files
    chosen_files = list(filter(which_files, raw_files))
    chosen_files.sort()
    for file_name in chosen_files:
        new_file_name = rename_out_files(file_name)
        path_out_file_with_name = folder_out_path + '/' + new_file_name
        func_to_run(folder_in_path + "/" + file_name, path_out_file_with_name)
    write_log(f'finish run_func_for_all with {func_to_run.__name__}')


def check_files_and_folder_for_complete_run(first_folder="./raw_data"):
    write_log('status: check_files_and_folder_for_complete_run: check missing default files and folder...')
    not_found = []
    folders = [first_folder, 'plots_folder1', 'plots_folder1/part2', 'plots_folder1/testing2_out', './csv_data2',
               './filtered_data3', './normalized_data4', './merged_data5', './clusttered_data6']
    for folder in folders:
        if not os.path.isdir(folder):
            print(f"did not found folder named: {folder}")
            not_found.append(folder + ' (folder)')

    chosen_files = os.listdir(first_folder)
    chosen_files.sort()
    if 'MEA_dimorphism_samples.xlsx' not in chosen_files:
        print(
            f'did not found \'MEA_dimorphism_samples.xlsx\' (file. assume it should be in \'./raw_dara\'')
        not_found.append(f'{first_folder}/MEA_dimorphism_samples.xlsx (file)')

    flag = False
    for file in chosen_files:
        if '_features.tsv' in file:
            flag = True
            break
    if not flag:
        print(
            f'did not found any  \'<num>_features.tsv\' (file. assume it should be in \'./raw_dara\'')
        not_found.append(f'{first_folder}/<num>_features.tsv (file)')

    mtx_files = list(filter(lambda x: '_matrix.mtx' in x, chosen_files))
    mtx_files.sort()
    barcodes_files = list(filter(lambda x: '_barcodes.tsv' in x, chosen_files))
    barcodes_files.sort()

    for file in mtx_files:
        num = file[:5]
        flag = False
        for barcode in barcodes_files:
            if num in barcode:
                flag = True
        if not flag:
            print(f"did not found barcode.tsv file for {file} (sample {num})")
            not_found.append(f'{first_folder}/{num}_barcode.tsv (file)')

    if len(mtx_files) != len(barcodes_files):
        print(
            f"numbers of mtx files and barcodes files are different! something is missing")
        not_found.append(
            f'numbers of mtx files and barcodes files are different! something is missing')

    if len(not_found) == 0:
        write_log("All files and folder are found! have a great flight!")
        return True
    else:
        print("Found there are some missing files and folders! better check them before running...")
        import sys
        time.sleep(0.3)
        print("Warning: there are some missing files and folders:", file=sys.stderr)
        for index, current in enumerate(not_found):
            print(f"{index+1}) {current}", file=sys.stderr)
        print("Warning: we are suggesting to create those files and folder before continuing", file=sys.stderr)

    return False


def stack_csv_together(folder_path, out_file_path='./merged_data5/stacked_mtx.csv'):
    write_log(f'start stack_csv_together')
    chosen_files = os.listdir(folder_path)  # list all raw files
    # print(raw_files)
    # raw_files = list(filter(lambda x: '_matrix2.csv' in x, raw_files))
    chosen_files = list(
        filter(lambda x: 'matrix_normalized.csv' in x, chosen_files))
    chosen_files.sort()
    print('status: stack_csv_together for', chosen_files)

    stacked_csv = pd.read_csv(
        folder_path + "/" + chosen_files[0], index_col=0, header=0)
    num_sample = chosen_files[0][:4]
    col_list = list(stacked_csv.columns)
    col_list = [str(x) + "__" + num_sample for x in col_list]
    stacked_csv.columns = col_list

    log_info = []
    sum_index = stacked_csv.shape[1] + 1
    log_info.append((chosen_files[0], 1))
    for file in chosen_files[1:]:
        tmp = pd.read_csv(folder_path + "/" + file, index_col=0, header=0)
        num_sample = file[:4]
        col_list = list(tmp.columns)
        col_list = [str(x)+"__"+num_sample for x in col_list]
        tmp.columns = col_list

        stacked_csv = pd.concat([stacked_csv, tmp], axis=1)

        log_info.append((file, sum_index))
        sum_index += tmp.shape[1]

    print(log_info)
    f = open(f'./ml_run_logs.txt', 'a+')
    msg = str(datetime.datetime.now()) + \
        " stack_csv_together: " + log_info.__str__() + "\n"
    f.write(msg)

    stacked_csv.to_csv(out_file_path)
    print('status: finish stack_csv_together. the new concat file called', out_file_path)


def merge_all_metadata(folder_path='./filtered_data3', out_file='./merged_data5/all_samples_metadata.csv'):
    chosen_files = os.listdir(folder_path)  # list all raw files
    chosen_files = list(
        filter(lambda x: '_metadata_filtered.csv' in x, chosen_files))
    chosen_files.sort()
    print('status: merge_all_metadata for', chosen_files)
    metadatas_csv = pd.read_csv(
        folder_path + "/" + chosen_files[0], index_col=0, header=0)
    for file in chosen_files[1:]:
        tmp = pd.read_csv(folder_path + "/" + file, index_col=0, header=0)
        metadatas_csv = pd.concat([metadatas_csv, tmp])
    metadatas_csv.to_csv(out_file)
    print('status: finish merge_all_metadata. the new concat file called', out_file)


def find_indices_of_gene(folder_path='./raw_csv_data2', gene_to_filter='mt-'):
    raw_files = os.listdir(folder_path)  # list all raw files
    chosen_files = list(filter(lambda x: 'features.csv' in x, raw_files))
    features_csv = pd.read_csv(
        folder_path + "/" + chosen_files[0], index_col=0, header=0)
    return np.array(features_csv[features_csv['geneName'].str.startswith(gene_to_filter)].index)


def split_merged_into_M_F(path_stacked_file='./merged_data5/stacked_normalized_filtered_threshold_mtx.csv', mea_samples='./raw_data/MEA_dimorphism_samples.xlsx', out_file_M='./merged_data5/stacked_M.csv', out_file_F='./merged_data5/stacked_F.csv'):
    print('Status: split_merged_into_M_F: start splitting the merged csv file into females and males csv')
    df_f_m_index = pd.read_excel(mea_samples)
    # print(df_f_m_index)
    f_list, m_list = [], []
    for index, row in df_f_m_index.iterrows():
        if row['female'] == 1:
            f_list.append(row.iloc[0])
        else:
            m_list.append(row.iloc[0])
    print('Females:', f_list)
    print('Males:', m_list)
    del df_f_m_index

    def filtering_gender(label_list, gender_list):
        indices = []
        for label in label_list:
            num = label.split('__')[1]
            if num in gender_list:
                indices.append(label)
        return indices

    df = pd.read_csv(path_stacked_file, index_col=0,
                     header=0, low_memory=False) #why not using the metadata
    merged_col_num = df.shape[1]
    print(
        f'Status: finish loading data (its shape: {df.shape}). now filtering only males')
    only_m = df.loc[:, filtering_gender(df.columns, m_list)]
    print(
        f'Status: created male df (shape {only_m.shape}). saving it into {out_file_M}')
    only_m.to_csv(out_file_M, sep=',')
    male_col_num = only_m.shape[1]
    del only_m

    print(f'Status: now filtering only females')
    only_f = df.loc[:, filtering_gender(df.columns, f_list)]
    print(
        f'Status: created female df (shape {only_f.shape}). saving it into {out_file_F}')
    only_f.to_csv(out_file_F, sep=',')
    female_col_num = only_f.shape[1]
    print(f'Status: finish splitting the merged csv file into females and males csv')

    if female_col_num + male_col_num != merged_col_num:
        import sys
        time.sleep(0.3)
        print(
            f"Warning: part of the cols could not be belong to any gender. for that reason, {merged_col_num - female_col_num - male_col_num} cols did not used neither of the male nor female new created csv files", file=sys.stderr)
        print(
            "Warning: we are suggesting check this out before continuing", file=sys.stderr)


def write_log(msg, print_std=True):
    f = open(f'./ml_run_logs.txt', 'a+')
    msg = str(datetime.datetime.now())[:-4] + ": " + msg
    if print_std:
        print(msg)
    f.write(msg + "\n")
    f.close()
