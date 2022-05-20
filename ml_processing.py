import pandas as pd
import numpy as np
import seaborn as sns
import os
import datetime
import matplotlib.pyplot as plt
from itertools import islice
from sklearn import decomposition
import data_plot_utils



def tSNE(path_in, path_out, plots_folder='./plots_folder1'):
    df = pd.read_csv(path_in, index_col=0, header=0)



