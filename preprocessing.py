import numpy as np
import pandas as pd


def neutralize_series(target, by, proportion=1.0):
    """ 線形回帰成分を控除して直交化する
    Args:
        target ([pd.Series]): 直交化対象の系列
        by ([pd.Series]): 直交化の際の横軸
        proportion (float, optional): 直交化の強さ. Defaults to 1.0.

    Returns:
        [pd.Series]: 直交化された系列
    """
    scores = target.values.reshape(-1, 1)
    factor = by.values.reshape(-1, 1)
    # データ行列 [n, 2]
    factor = np.hstack(
        (factor, np.array([np.mean(target)] * len(factor)).reshape(-1, 1))
    )
    # 横軸の値がx=byの時の回帰直線の値(ax+b)をtargetから控除
    exposure = np.linalg.lstsq(factor, scores, rcond=None)[0]
    correction_term = proportion * (factor.dot(exposure))
    neutralized_target = target.values.reshape(-1, 1) - correction_term
    neutralized_target = pd.Series(
        neutralized_target.reshape(-1,), index=target.index
    )
    return neutralized_target
