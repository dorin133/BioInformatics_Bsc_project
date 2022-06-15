import pandas as pd
import numpy as np
import numpy as np
# from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
import pandas as pd
import os

def erase_all_files_from_root(path_to_root_folder):
    # keep all the sub_directories, but empty them also 
    ch = input("Do you want to erase all output from previous experiments? [Y / N]\n")
    legal_input = ['y','Y','n','N']
    while (ch not in legal_input):
        print("Illegal input! try again...(enter only one letter, y or n) \n")
        ch = input("Do you want to erase all output from previous experiments? [Y / N]\n")
    if(ch == 'y' or ch == 'Y'):
        sub_dir_list = [path_to_root_folder]
        while sub_dir_list:
            dir = sub_dir_list.pop(0)
            for f in os.listdir(dir):
                # print(f)
                if(os.path.isfile(os.path.join(dir, f))):
                    os.remove(os.path.join(dir, f))
                else:
                    sub_dir_list.append(os.path.join(dir, f))

def check_for_marker_genes():
    x = [[5,21,101], [10,22,102], [15,25,103]]
    df = pd.DataFrame(x, columns=[1500,2000,3000])
    df_mean  = df.T.mean(axis=1)
    df_normalized = (df.T.sub(df_mean, axis=0))
    marker_genes_dict = {}
    selected_mareker_genes = [2000]
    for clustter_idx in df_normalized.columns:
        marker_genes_curr = df_normalized[clustter_idx][df_normalized[clustter_idx] > 0].nlargest(2)
        marker_genes_curr_genes = marker_genes_curr.index.to_list()
        marker_genes_dict[clustter_idx] = marker_genes_curr_genes
        # how to subtract one list from another: set(marker_genes_curr_genes) - set(selected_mareker_genes)
    pass


def exp4():
    print("exp4")
    df = pd.read_csv('./merged_data5/stacked_1.csv', index_col=0, header=0).T
    print('df.shape', df.shape)
    count = (df[1868] != 0).sum()
    print(count)
    count = (df[12665] != 0).sum()
    print(count)
    count = (df[12788] != 0).sum()
    print(count)
    print('end of exp4')

if __name__=="__main__":
    # dir = 'C:/Users/Dorin Shteyman/Documents/GitHub/brains_v2/test_folder' #path to root dir
    # erase_all_files_from_root(dir)
    # check_for_marker_genes()
    exp4()