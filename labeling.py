import datetime as dt
import numpy as np
import pandas as pd
import numba


def tripleBarrier(events, midPrice,
                  theta_ver=dt.timedelta(seconds=60), theta_hor=100,
                  barrier_type=[1,1,1]):
    """
    tripleBarrier法によるシグナルイベントt0のラベリング.
    Args:
        events (Series or DatetimeIndex): event timestamp t0.
        midPrice (pd.DF or Series with DatetimeIndex):
        theta_hor (float): threshold of uppper/lower barrier
        theta_ver (timedelta): threshold of vertical barrier
        barrier_type (list): multiple factor of each barriers ...[upper,lower,vertical]
    """
    assert "M8" in events.dtype.str, "events must be DatetimeIndex or Series contains Datetime."
    assert "M8" in midPrice.index.dtype.str, "midPrice must have DatetimeIndex."
    _midPrice = midPrice.sort_index().reset_index().values
    _midPrice[:,0] = midPrice.index.values.astype(float)/1e9
    _midPrice = _midPrice.astype(float)
    _events = events.values.astype(float)/1e9
    _events, t1, ret =\
        __tripleBarrier(_events, _midPrice, theta_ver.total_seconds(), theta_hor, np.array(barrier_type))
    result = pd.DataFrame({
        "t0" : pd.to_datetime((_events*1e9)).round("ms").values,
        "t1" : pd.to_datetime((t1*1e9)).round("ms").values,
        "ret": ret
    })
    result.t1.replace(dt.date(1970,1,1), pd.NaT, inplace=True)
    result.loc[result.t1.isna(), "ret"] = np.nan
    return result #pd.DF[t0,t1,return]


@numba.jit(nopython=True)
def __tripleBarrier(events, midPrice, theta_ver, theta_hor, barrier_type):

    t1  = np.zeros_like(events) #barrier接触時の時間
    ret = np.zeros_like(events) #barrier接触時のリターン
    mid0 = midPrice[0][1]
    _j = 0; J = len(midPrice)
    for i in range(len(events)):
        for j in range(_j, J):
            if midPrice[j][0] <= events[i]:
                #イベント以前
                _j=j; mid0=midPrice[j][1]
            elif events[i] < midPrice[j][0]:
                #イベント以後
                dt_now  = (midPrice[j][0] - events[i])
                ret_now = (midPrice[j][1] - mid0)
                if   barrier_type[0]>0 and ret_now >= +theta_hor*barrier_type[0]: #upper
                    t1[i]  = midPrice[j][0]
                    ret[i] = ret_now
                    break
                elif barrier_type[1]>0 and ret_now <= -theta_hor*barrier_type[1]: #lower
                    t1[i]  = midPrice[j][0]
                    ret[i] = ret_now
                    break
                elif barrier_type[2]>0 and dt_now > theta_ver*barrier_type[2]: #vertical
                    t1[i]  = midPrice[j][0]
                    ret[i] = ret_now
                    break
            elif j == J-1:
                #末尾に辿り着いてしまった
                t1[i]  = np.nan
                ret[i] = np.nan
    return events, t1, ret


def sampleUniqness(events, time_interval):
    """
    サンプルの平均独自性(0<.<1)を計算
    Args:
        events (Series or DataTimeindex): sampling bar t
        time_interval (pd.DF): 各リターン区間iの[始点,終点]
    """
    assert "M8" in events.values.dtype.str,\
        "events must be DatetimeIndex or Series of Datetime"
    assert "M8" in time_interval.iloc[:,0].dtype.str and type(time_interval)==pd.DataFrame,\
        "time_interval must have DatetimeIndex and is pd.DataFrame"
    idx = events.values.astype(float)/1e9
    itv = time_interval.values.astype(float)/1e9
    mean_uniqness = __sampleUniqness(idx, itv)
    mean_uniqness = (time_interval).join(pd.DataFrame(mean_uniqness, columns=["uniqness"]))
    return mean_uniqness


@numba.jit(nopython=True)
def __numBarInRet(time_index, time_interval):
    # 各リターンがbarをいくつ含むかを計算(mat.sum(axis=0))
    count_arr = np.zeros(time_interval.shape[0])
    _i = 0; I = len(time_index)
    for j in range(len(time_interval)): #return sample loop:j
        for i in range(_i, I): #bar loop:i
            if time_index[i] < time_interval[j,0]:
                _i = i
            elif time_index[i] <= time_interval[j,1]:
                count_arr[j] += 1
            elif time_interval[j,1] < time_index[i]:
                break
    return count_arr

@numba.jit(nopython=True)
def __c_t(time_index, time_interval):
    # 各barがリターンに何回現れるかを計算(mat.sum(axis=1))
    # __sampleUniqnessで使用.
    count_arr = np.zeros(time_index.shape[0])
    _i = 0; I = len(time_index)
    for j in range(len(time_interval)): #return sample loop:j
        for i in range(_i, I): #bar loop:i
            if time_index[i] < time_interval[j,0]:
                _i = i
            elif time_index[i] <= time_interval[j,1]:
                count_arr[i] += 1
            elif time_interval[j,1] < time_index[i]:
                break
    return count_arr

@numba.jit(nopython=True)
def __sampleUniqness(time_index, time_interval):
    # リターンiの平均独自性を計算
    uniqness = np.zeros(time_interval.shape[0])
    c_t = __c_t(time_index, time_interval) #各barのリターン中での出現回数:index=i
    n_b = __numBarInRet(time_index, time_interval) #各リターン中でのbarの出現回数:index=j
    _i = 0; I = len(time_index)
    for j in range(len(time_interval)): #return sample loop
        for i in range(_i, I): #bar loop
            if time_index[i] < time_interval[j,0]:
                _i = i
            elif time_index[i] <= time_interval[j,1]:
                uniqness[j] += 1 / c_t[i]
            elif time_interval[j,1] < time_index[i]:
                break
    return uniqness / n_b


def integratedReturn(events, midPrice, theta_ver):
    """ 時間で積分されたリターンを返す.
    Args:
        events (Series or DatetimeIndex): sampled event timestamps.
        midPrice (pd.DF or Series with DatetimeIndex):
        theta_ver (timedelta): threshold of vertical barrier
    """
    assert "M8" in events.dtype.str, "events must be DatetimeIndex."
    assert "M8" in midPrice.index.dtype.str, "midPrice must have DatetimeIndex."
    _midPrice = midPrice.sort_index().reset_index().values
    _midPrice[:,0] = midPrice.index.values.astype(float)/1e9
    _midPrice = _midPrice.astype(float) # [N,2]
    _events = events.values.astype(float)/1e9  # [n,]
    ret = __integratedReturn(_events, _midPrice, theta_ver.total_seconds())
    result = pd.DataFrame({
        "t0" : pd.to_datetime((_events*1e9)).round("ms").values,
        "t1" : events+theta_ver,
        "ret": ret
    })
    return result

@numba.jit(nopython=True)
def __integratedReturn(events, midPrice, theta_ver):
    # 時間で積分したリターンを返す.
    _t = 0
    _price_init = -1 #initial price of sample
    _price_last = -1
    _time_last = -1 #unixtime
    T = max(midPrice[:,0])
    ret = np.zeros(events.shape[0])
    for i in range(len(ret)): #return sample loop:i
        for t in range(_t, len(midPrice)): #time sample loop:t
            if T < events[i]+theta_ver:
                ret[i] = np.nan #リターンが計算可能な時間外の場合nan
                break
            elif midPrice[t,0] < events[i]:
                pass
            elif events[i] <= midPrice[t,0] and midPrice[t,0] < events[i]+theta_ver:
                if _price_init < 0:
                    _time_last  = midPrice[t,0]
                    _price_init = _price_last = midPrice[t,1]
                    _t = max(t-1,0)
                ret[i] += (_price_last - _price_init) * (midPrice[t,0] - _time_last)
                _time_last = midPrice[t,0]
                _price_last = midPrice[t,1]
            else:
                _time_last = -1
                _price_init = -1
                _price_last = -1
                break
    return ret