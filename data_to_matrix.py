import pandas as pd
import numpy as np
import os

def mtx_to_pandas(file_path, path_out):
    f_input = open(file_path, 'r')
    f_input.readline()
    f_input.readline()
    line = f_input.readline()
    col_size = int(line.split(' ')[1])
    tmp = pd.DataFrame(0, index=[i for i in range(1, 27999)], columns=[i for i in range(1, col_size+1)])
    index = 0
    for line in f_input:
        if len(line) > 3:
            curr_feature, curr_barcode, curr_value = line[:-1].split(' ')
            curr_feature, curr_barcode, curr_value = int(curr_feature), int(curr_barcode), int(curr_value)
            tmp.iloc[curr_feature-1, curr_barcode-1] = curr_value

            print(index)
            index += 1
    f_input.close()
    tmp.to_csv(path_out)


def mtx_to_numpy_csv(file_path, path_out):
    f_input = open(file_path, 'r')
    f_input.readline()
    f_input.readline()
    line = f_input.readline()
    col_size = int(line.split(' ')[1])
    table = np.zeros((27999, col_size+1), dtype=int)
    index = 0
    for line in f_input:
        if len(line) > 3:
            curr_feature, curr_barcode, curr_value = line[:-1].split(' ')
            curr_feature, curr_barcode, curr_value = int(curr_feature), int(curr_barcode), int(curr_value)
            table[curr_feature-1, curr_barcode-1] = curr_value

            print(index)
            index += 1
            if index % 10000 == 0:
                print(index)
                # break
    f_input.close()
    df = pd.DataFrame(table)
    df.drop([0], inplace=True, axis=0)
    df.drop([0], inplace=True, axis=1)
    df.to_csv(path_out)
    # tmp.tofile(path_out, sep = ",")

def all_mtx_to_pandas(folder_path, path_out_folder):
    raw_files = os.listdir(folder_path)  # list all raw files
    # print(raw_files)
    raw_files = list(filter(lambda x: '_matrix.mtx' in x, raw_files))  # filter files which are not barcodes files
    # print(raw_files)

    for mtx in raw_files:
        sample_num = mtx[-15:-7]
        path_out_file = path_out_folder + '/' + sample_num + "_matrix2"
        mtx_to_pandas(mtx, path_out_file)


def stack_all_csv_together(folder_path):
    raw_files = os.listdir(folder_path)  # list all raw files
    # print(raw_files)
    raw_files = list(filter(lambda x: '_matrix2.csv' in x, raw_files))  # filter files which are not barcodes files
    # print(raw_files)

    stacked_mtx = pd.read_csv(raw_files[0])
    for mtx in raw_files[1:]:
        stacked_mtx = pd.concat([stacked_mtx, mtx], axis=1, )



if __name__ == '__main__':
    mtx_to_numpy_csv('./raw_data/35_1_matrix.mtx', './raw_data/mtx22.csv')
    print("hi")
