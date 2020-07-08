import numpy as np
import pandas as pd
import time,datetime
import numba

@numba.jit(nopython=True)
def drawDown(pnl:np.array)->np.array:
    """ 各時点tのdraw down.
    Args:
        pnl (ndarray): 
    """
    T = pnl.shape[0]
    DD = np.zeros_like(pnl)
    for t in range(T-1):
        if pnl[t] > pnl[t+1]:
            for t0 in range(t,T-1):
                DD[t] = min(pnl[t0]-pnl[t], DD[t])
                if pnl[t]<pnl[t0]: break
    return DD #ndarray


def aftermath(midprice, signal, lookforward):
    aftermath = np.concatenate([
                    pd.merge_asof(pd.DataFrame(index=signal+datetime.timedelta(seconds=n)), 
                    midprice.to_frame(), left_index=True,right_index=True,direction='forward').values
                    for n in range(lookforward)], axis=1)
    aftermath = aftermath - aftermath[:,0].reshape(-1,1)
    return pd.DataFrame(aftermath, columns=[n for n in range(lookforward)], index=signal)