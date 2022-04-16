import os
import time

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


def check_files_and_folder_for_complete_run(first_folder="./raw_data"):
    print('status: check_files_and_folder_for_complete_run: check missing default files and folder...')
    not_found = []
    folders = [first_folder, 'plots_folder1', './csv_data2', './filtered_data3', './normalized_data4', './merged_data5']
    for folder in folders:
        if not os.path.isdir(folder):
            print(f"did not found folder named: {folder}")
            not_found.append(folder + ' (folder)')

    chosen_files = os.listdir(first_folder)
    chosen_files.sort()
    if 'MEA_dimorphism_samples.xlsx' not in chosen_files:
        print(f'did not found \'MEA_dimorphism_samples.xlsx\' (file. assume it should be in \'./raw_dara\'')
        not_found.append(f'{first_folder}/MEA_dimorphism_samples.xlsx (file)')

    flag = False
    for file in chosen_files:
        if '_features.tsv' in file:
            flag = True
            break
    if not flag:
        print(f'did not found any  \'<num>_features.tsv\' (file. assume it should be in \'./raw_dara\'')
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
        print(f"numbers of mtx files and barcodes files are different! something is missing")
        not_found.append(f'numbers of mtx files and barcodes files are different! something is missing')

    if len(not_found) == 0:
        print("All files and folder are found! have a great flight!")
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


def merge_all_metadata(folder_path='./filtered_data3', out_file='./merged_data5/all_samples_metadata.csv'):
    chosen_files = os.listdir(folder_path)  # list all raw files
    chosen_files = list(filter(lambda x: '_metadata_filtered.csv' in x, chosen_files)) 
    chosen_files.sort()
    print('status: merge_all_metadata for', chosen_files)
    metadatas_csv = pd.read_csv(folder_path + "/" + chosen_files[0], index_col=0, header=0)
    for file in chosen_files[1:]:
        tmp = pd.read_csv(folder_path + "/" + file, index_col=0, header=0)
        metadatas_csv = pd.concat([metadatas_csv, tmp])
    metadatas_csv.to_csv(out_file)
    print('status: finish merge_all_metadata. the new concat file called', out_file)


def find_indices_of_gene(folder_path='./raw_csv_data2', gene_to_filter='mt-'):
    raw_files = os.listdir(folder_path)  # list all raw files
    chosen_files = list(filter(lambda x: 'features.csv' in x, raw_files))
    features_csv = pd.read_csv(folder_path + "/" + chosen_files[0], index_col=0, header=0)
    return np.array(features_csv[features_csv['geneName'].str.startswith(gene_to_filter)].index)


