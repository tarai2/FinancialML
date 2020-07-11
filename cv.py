import datetime
import numpy as np
import pandas as pd
import numba
from typing import List


def purge(data, k=10):
    """ リターンの重複を許さずにデータセットを分割
    Args:
        data (pd.DF): [int| t0,t1,ret,weight]
        k (int):
    """
    N_row = data.shape[0]
    kfold_row = int(N_row / k)
    temp_time = data.t0.min()
    purged_folds = []
    for i in range(k):
        temp_fold =\
             data.iloc[kfold_row*i: kfold_row*(i+1)].query("@temp_time <= t0")
        if temp_fold.shape[0] > 0:
            purged_folds.append(temp_fold)
        temp_time = temp_fold.t1.max()
    return purged_folds


def cross_validation(model, folded_data: List):
    pass


def adv_validation():
    pass
