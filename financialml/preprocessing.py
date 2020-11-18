import numpy as np
import pandas as pd
from scipy.special import erfinv


def neutralizeSeries(target, by, proportion=1.0):
    """ 線形回帰成分を控除してtargetをbyに対して直交化する
    Args:
        target (pd.Series): 直交化対象の系列
        by (pd.Series): 直交化の際の横軸
        proportion (float, optional): 直交化の強さ. Defaults to 1.0.
    Returns:
        pd.Series: 直交化された系列
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


def neutralize(target, by, proportion=1.0):
    """ 線形回帰成分を控除してtargetをbyに対して直交化する
    Args:
        target (pd.Series): 直交化対象の系列
        by (pd.DataFrame): 直交化の際の横軸
        proportion (float, optional): 直交化の強さ. Defaults to 1.0.
    Returns:
        pd.Series: 直交化された系列
    """
    factors = by.values
    # constant column to make sure the series is completely neutral to exposures
    factors = np.hstack(
        (factors,
         np.asarray(np.mean(target)) * np.ones(len(factors)).reshape(-1, 1))
    )

    scores = target - proportion * factors.dot(np.linalg.pinv(factors).dot(target.values))
    return scores / scores.std()


def applyRankGauss(target):
    """ 特徴量target(離散,連続)をrankgauss化
    Args:
        target (pd.Series):
    Returns:
        pd.Series:
    """
    ranking = target.rank()
    ranking_pm1 = 2 * (ranking / (ranking.max()+1) - 0.5)
    rankgaussed_series = erfinv(ranking_pm1)
    return rankgaussed_series


def applyBasedRankGauss(target, base=0, method="min"):
    """ 特徴量target(離散,連続)をrankgauss化. baseの値を正規化後の0と一致させる.
    Args:
        target (pd.Series):
    Returns:
        pd.Series:
    """
    indicator_U = (target>base)
    indicator_D = (target<base)
    indicator_N = (target == base)
    target[indicator_U] = +target[indicator_U].rank() / (indicator_U.sum()+1)
    target[indicator_D] = -target[indicator_D].rank() / (indicator_D.sum()+1)
    target[indicator_N] = 0.
    rankgaussed = erfinv(target)
    return rankgaussed