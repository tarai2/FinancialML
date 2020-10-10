import numpy as np
import pandas as pd


def get_dummied1d(array_1d, threshold=[0, 0]):
    """ 1d-float -> onehot-UND
    Args:
        array_1d (pd.DataFrame): 予測値(float)の1d-array.
        threshold (list of float):
    Returns:
        (pd.DataFrame): 
    """
    assert hasattr(array_1d, "index"), "array_1d must be pd.Series or DataFrame."
    arr_labeled = to3classLabel(array_1d, threshold)
    arr_onehot = pd.get_dummies(arr_labeled)
    return arr_onehot


def get_und_dummies(arr, **kwargs):
    """ 1d-UND -> onehot-UND
    """
    dummied = pd.get_dummies(arr, **kwargs)
    if dummied.columns.shape[0] > 3:
        raise ValueError("dummied label contains other than U, D or N.")
    elif dummied.columns.shape[0] < 3:
        dropped = set(["U","N","D"]) - set(dummied.columns)
        for col in dropped:
            dummied[col] = np.zeros_like(dummied.iloc[:,0])
        dummied = dummied.sort_index(axis=1)
    return dummied
