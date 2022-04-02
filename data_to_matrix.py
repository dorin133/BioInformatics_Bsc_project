import pandas as pd
import numpy as np
import os

# def mtx_to_pandas(file_path, path_out):
#     f_input = open(file_path, 'r')
#     f_input.readline()
#     f_input.readline()
#     line = f_input.readline()
#     col_size = int(line.split(' ')[1])
#     tmp = pd.DataFrame(0, index=[i for i in range(1, 27999)], columns=[i for i in range(1, col_size+1)])
#     index = 0
#     for line in f_input:
#         if len(line) > 3:
#             curr_feature, curr_barcode, curr_value = line[:-1].split(' ')
#             curr_feature, curr_barcode, curr_value = int(curr_feature), int(curr_barcode), int(curr_value)
#             tmp.iloc[curr_feature-1, curr_barcode-1] = curr_value

#             print(index)
#             index += 1
#     f_input.close()
#     tmp.to_csv(path_out)


def mtx_to_numpy_csv(file_path, path_out):
    print(f'status: start proccessing {file_path}')
    f_input = open(file_path, 'r')
    f_input.readline()
    f_input.readline()
    line = f_input.readline()
    col_size = int(line.split(' ')[1])
    table = np.zeros((27999, col_size+1), dtype=int)
    for index, line in enumerate(f_input):
        if len(line) > 3:  # only read valid lines
            curr_feature, curr_barcode, curr_value = line[:-1].split(' ')  # assuming already know the format
            curr_feature, curr_barcode, curr_value = int(curr_feature), int(curr_barcode), int(curr_value)
            table[curr_feature, curr_barcode] = curr_value

            if index % 10000 == 0:
                print(f'status: reached line #{index}')
                
    f_input.close()
    print(f'status: finish proccessing mtx file. creating csv file')
    df = pd.DataFrame(table)
    df.drop([0], inplace=True, axis=0)
    df.drop([0], inplace=True, axis=1)
    df.to_csv(path_out)
    print(f'status: created the file "{path_out}"')
    # tmp.tofile(path_out, sep = ",")


def all_mtx_to_pandas(folder_path="./raw_data", path_out_folder="./raw_data"):
    raw_files = os.listdir(folder_path)  # list all raw files
    # print(raw_files)
    raw_files = list(filter(lambda x: '_matrix.mtx' in x, raw_files))  # filter files which are not barcodes files
    # print(raw_files)

    for mtx in raw_files:
        sample_num = mtx[-15:-11]
        path_out_file = path_out_folder + '/' + sample_num + "_matrix2.csv"
        mtx_to_numpy_csv(mtx, path_out_file)


def stack_all_csv_together(folder_path, out_file_path='./parsed_data/stacked_mtx.csv'):
    raw_files = os.listdir(folder_path)  # list all raw files
    # print(raw_files)
    raw_files = list(filter(lambda x: '_matrix2.csv' in x, raw_files))  # filter files which are not barcodes files
    print(raw_files)

    stacked_mtx = pd.read_csv(folder_path + "/" + raw_files[0], index_col=0, header=0, dtype=np.int32)
    for mtx in raw_files[1:]:
        another_mtx = pd.read_csv(folder_path + "/" + mtx, index_col=0, header=0, dtype=np.int32)
        stacked_mtx = pd.concat([stacked_mtx, another_mtx], axis=1)

    stacked_mtx.to_csv(out_file_path)


########################################################################################################################


def prepare_single_files(folder_path='./raw_data', out_folder_path='./raw_data'):
    raw_files = os.listdir(folder_path)  # list all raw files
    # print(raw_files)
    raw_files = list(filter(lambda x: '_barcodes.tsv' in x, raw_files))  # filter files which are not barcodes files
    # print(raw_files)
    dataset  = pd.read_excel('./raw_data/MEA_dimorphism_samples.xlsx', names =['sample_id', 'female', 'parent'],index_col=None)
    for file_name in raw_files:
        tmp = file_name.index('_barcodes')
        file_id = file_name[:tmp]
        input_file_path = folder_path + "/" + file_name
        f_input = open(input_file_path, 'r')
        output_file_path = out_folder_path + "/" +file_id +'_metadata' + '.csv'
        f_output = open(output_file_path, 'a+')
        line_to_write_female = str(dataset[dataset['sample_id']==file_id].female.unique()[0])
        line_to_write_parent = str(dataset[dataset['sample_id']==file_id].parent.unique()[0])
        for line in f_input:
            line_to_write = line[:-3] + '_' + file_id + ' '+ file_id
            final_line  = line_to_write + ' ' + line_to_write_female + ' ' + line_to_write_parent + '\n'
            f_output.write(final_line)
        f_input.close()
        f_output.close()


def merge_all_files(folder_path='./raw_data', out_folder_path='./raw_data'):
    raw_files = os.listdir(folder_path)  # list all raw files
    # print(raw_files)
    raw_files = list(filter(lambda x: '_metadata.csv' in x, raw_files))  # filter files which are barcodes files
    # print(raw_files)

    output_file_path = out_folder_path + "/all_samples_barcodes.csv"
    f_output = open(output_file_path, 'a+')
    for file_name in raw_files:
        input_file_path = folder_path + "/" + file_name
        f_input = open(input_file_path, 'r')
        f_output.write(f_input.read())
        f_input.close()
    f_output.close()
    return  # TODO


if __name__ == '__main__':
    # prepare_single_files()
    # merge_all_files()
    
    mtx_to_numpy_csv('./raw_data/tmp2.mtx', './parsed_data/tmp22.csv')

    # all_mtx_to_pandas()
    # stack_all_csv_together('./parsed_data')
    
    print("Done")
