import datetime
import time
import numba
import numpy as np
import pandas as pd


@numba.jit(nopython=True)
def drawDown(pnl: np.array) -> np.array:
    """ 各時点tのdraw down.
    Args:
        pnl (ndarray):
    Returns:
        ndarray: max draw down at time i
    """
    T = pnl.shape[0]
    DD = np.zeros_like(pnl)
    for t in range(T-1):
        if pnl[t] > pnl[t+1]:
            for t0 in range(t, T-1):
                DD[t] = min(pnl[t0]-pnl[t], DD[t])
                if pnl[t] < pnl[t0]:
                    break
    return DD  # ndarray


def getAftermath(midprice, signal, lookforward):
    """ midpriceのSeriesからsignalのAftermathを計算
    Args:
        midprice (pd.Series):
        signal (DatetimeIndex):
        lookforward (int): seconds
    Returns:
        pd.DataFrame: [signalDate|aftermath_0,...,aftermath_n]
    """

    midprice_df = midprice.to_frame()
    # ndarray[sample, deltaTime]を作成
    aftermath = np.concatenate(
        [
            pd.merge_asof(
                pd.DataFrame(index=signal+datetime.timedelta(seconds=n)),
                midprice_df,
                left_index=True,
                right_index=True,
                direction='forward'
            ).values
            for n in range(lookforward)
        ],
        axis=1
    )
    # 初期値を引く
    aftermath = aftermath - aftermath[:, 0].reshape(-1, 1)

    return pd.DataFrame(
        aftermath,
        columns=[n for n in range(lookforward)],
        index=signal
    )
