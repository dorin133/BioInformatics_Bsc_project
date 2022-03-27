import pandas as pd
import os

# Each sample has number such as 52_1 and there are three files per sample.
# 1. barcodes.tsv which is the cell ID
# 2. features.tsv which is the gene name --> all files are identical
# 3. matrix.mtx which is the data matrix in mtx format --> first col is the gen number (from feature file), the second col is the num of the "sample" it was taken from, and we assume that the third col is how much was of this gen.
# There is an Excel file with the list of samples and flags of female(0/1), parent(0/1).

# The first step is to load the data and merge it while keeping the cell ID and metadata information.
# Try to write function that load and merge the data into one big matrix.
# The cell ID (barcode.tsv) file is not 100% unique when merging the samples so add to the cell barcode the sample name for example ATATAGABAGA_52_1


def add_id_to_barcodes(folder_path='./raw_data', out_folder_path='./parsed_data'):
    raw_files = os.listdir(folder_path)  # list all raw files
    # print(raw_files)
    raw_files = list(filter(lambda x: '_barcodes.tsv' in x, raw_files))  # filter files which are not barcodes files
    # print(raw_files)
    for file_name in raw_files:
        tmp = file_name.index('_barcodes')
        file_id = file_name[:tmp]
        input_file_path = folder_path + "/" + file_name
        f_input = open(input_file_path, 'r')
        output_file_path = out_folder_path + "/" + file_name[:-4] + '_v2' + '.tsv'
        f_output = open(output_file_path, 'a+')
        for line in f_input:
            line_to_write = line[:-3] + '_' + file_id + '\n'
            f_output.write(line_to_write)
        f_input.close()
        f_output.close()


def load_features_dict(file_path='./raw_data/35_1_features.tsv'):
    features = {}
    f_input = open(file_path, 'r')
    for index, line in enumerate(f_input):
        split_line = line.split('\t')
        features[index+1] = split_line[0] + " " + split_line[1]
    f_input.close()
    return features


def load_barcodes_dict(file_path):
    barcodes = {}
    f_input = open(file_path, 'r')
    for index, line in enumerate(f_input):
        barcodes[index+1] = line[:-1]
    f_input.close()
    return barcodes


def matrix_per_sample(matrix_path, barcode_path, output_file, features: dict = load_features_dict()):
    barcodes = load_barcodes_dict(barcode_path)
    f_input = open(matrix_path, 'r')
    f_output = open(output_file, 'a+')
    for _ in range(3):
        f_output.write(f_input.readline())
    for line in f_input:
        curr_feature = int(line.split(' ')[0])
        curr_barcode = int(line.split(' ')[1])
        line_to_write = line[:-1] + ' ' + features[curr_feature] + ' ' + barcodes[curr_barcode] + '\n'
        f_output.write(line_to_write)
    f_input.close()
    f_output.close()


def merge_matrix_barcode_features(folder_path='./raw_data', out_folder_path='./parsed_data', features: dict = load_features_dict()):
    raw_files = os.listdir(folder_path)  # list all raw files
    # print(raw_files)
    raw_files = list(filter(lambda x: '_matrix.mtx' in x, raw_files))  # filter files which are not barcodes files
    # print(raw_files)
    for file_name in raw_files[:3]:
    # for file_name in raw_files:
        out_name = out_folder_path + "/" + file_name[:-4] + "_merged_with_barcode.mtx"
        file_path = folder_path + "/" + file_name
        barcode_path = folder_path + "/" + file_name[:-11] + "_barcodes_v2.tsv"
        matrix_per_sample(file_path, barcode_path, out_name, features)


def merge_all_files(folder_path='./parsed_data', out_folder_path='./parsed_data'):
    raw_files = os.listdir(folder_path)  # list all raw files
    # print(raw_files)
    raw_files = list(filter(lambda x: '_matrix_merged_with_barcode.mtx' in x, raw_files))  # filter files which are not barcodes files
    # print(raw_files)

    output_file_path = out_folder_path + "/all_samples_matrix.mtx.tsv"
    f_output = open(output_file_path, 'a+')

    flag_first = True
    line_counter = 0
    biggest_second_col = 0
    for file_name in raw_files:
        input_file_path = folder_path + "/" + file_name
        f_input = open(input_file_path, 'r')
        if flag_first:
            f_output.write(f_input.readline())
            f_output.write(f_input.readline())
        else:
            f_input.readline()
            f_input.readline()

        split_line = f_input.readline().split(' ')
        curr_second_col = int(split_line[1])
        if curr_second_col > biggest_second_col:
            biggest_second_col = curr_second_col
        line_counter += int(split_line[2])

        f_output.write(f_input.read())
        f_input.close()
    f_output.close()

    return  # TODO

    output_file_path_2 = out_folder_path + "/all_sampels_matrix_fix2.mtx.tsv"
    f_output2 = open(output_file_path_2, 'a+')
    f_output1 = open(output_file_path, 'r')

    f_output2.write(f_output1.readline())
    f_output2.write(f_output1.readline())
    # f_output1.readline()
    f_output2.write(f"27998 {biggest_second_col} {line_counter}")
    f_output2.write(f_output1.read())





if __name__ == '__main__':
    # add_id_to_barcodes()
    # features = load_features_dict()
    # bar = load_barcodes_dict('./raw_data/35_1_barcodes.tsv')
    # matrix_per_sample("./raw_data/35_1_matrix.mtx", )
    # matrix_per_sample("./raw_data/35_1_matrix.mtx", './raw_data/35_1_barcodes.tsv', output_file='./matrix_merged.mtx')
    # merge_matrix_barcode_features()
    merge_all_files()
    print("Done")
