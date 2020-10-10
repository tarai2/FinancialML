import datetime as dt
import numpy as np
import pandas as pd
import numba
import warnings
from .conversion import *


def getTripleBarrier(events, midPrice,
                     theta_ver=dt.timedelta(seconds=60), theta_hor=100,
                     barrier_type=[1, 1, 1]):
    """
    tripleBarrier法によるシグナルイベントt0のラベリング.
    Args:
        events (Series or DatetimeIndex): event timestamp t0.
        midPrice (pd.DF or Series with DatetimeIndex):
        theta_hor (float): threshold of uppper/lower barrier
        theta_ver (timedelta): threshold of vertical barrier
        barrier_type (list): multiple factor of each barriers ...[upper,lower,vertical]
    Return:
        pd.DataFrame: [start_date, end_date, return]
    """
    assert "M8" in events.dtype.str,\
           "events must be DatetimeIndex or Series contains Datetime."
    assert "M8" in midPrice.index.dtype.str,\
           "midPrice must have DatetimeIndex."

    _midPrice = midPrice.sort_index().reset_index().values
    _midPrice[:, 0] = midPrice.index.values.astype(float)/1e9
    _midPrice = _midPrice.astype(float)
    _events = events.values.astype(float)/1e9
    _events, t1, ret =\
        __tripleBarrier(
            _events,
            _midPrice,
            theta_ver.total_seconds(),
            theta_hor,
            np.array(barrier_type)
        )

    result = pd.DataFrame({
        "t0": pd.to_datetime((_events*1e9)).round("ms").values,
        "t1": pd.to_datetime((t1*1e9)).round("ms").values,
        "ret": ret
    })
    result.t1.replace(dt.date(1970, 1, 1), pd.NaT, inplace=True)
    result.loc[result.t1.isna(), "ret"] = np.nan
    return result  # pd.DF[t0,t1,return]


@numba.jit(nopython=True)
def __tripleBarrier(events, midPrice, theta_ver, theta_hor, barrier_type):

    t1 = np.zeros_like(events)  # barrier接触時の時間
    ret = np.zeros_like(events)  # barrier接触時のリターン
    mid0 = midPrice[0][1]
    _j = 0
    J = len(midPrice)
    for i in range(len(events)):
        for j in range(_j, J):
            if midPrice[j][0] <= events[i]:
                # イベント以前
                _j = j
                mid0 = midPrice[j][1]
            elif events[i] < midPrice[j][0]:
                # イベント以後
                dt_now = (midPrice[j][0] - events[i])
                ret_now = (midPrice[j][1] - mid0)
                if barrier_type[0] > 0\
                   and ret_now >= theta_hor * barrier_type[0]:  # upper
                    t1[i] = midPrice[j][0]
                    ret[i] = ret_now
                    break
                elif barrier_type[1] > 0 and ret_now <= -theta_hor * barrier_type[1]:  # lower
                    t1[i] = midPrice[j][0]
                    ret[i] = ret_now
                    break
                elif barrier_type[2] > 0 and dt_now > theta_ver*barrier_type[2]:  # vertical
                    t1[i] = midPrice[j][0]
                    ret[i] = ret_now
                    break
            elif j == J-1:
                # 末尾に辿り着いてしまった
                t1[i] = np.nan
                ret[i] = np.nan
    return events, t1, ret


def getSampleUniqness(events, time_interval):
    """ サンプルの平均独自性(0<.<1)を計算
    Args:
        events (Series or DataTimeindex): sampling bar t
        time_interval (pd.DF): 各リターン区間iの[始点,終点]
    Returns:
        pd.DataFrame: [start_date, end_date, uniqness]
    """
    assert "M8" in events.values.dtype.str,\
        "events must be DatetimeIndex or Series of Datetime"
    assert "M8" in time_interval.iloc[:, 0].dtype.str and type(time_interval) == pd.DataFrame,\
        "time_interval must have DatetimeIndex and is pd.DataFrame"
    idx = events.values.astype(float)/1e9
    itv = time_interval.values.astype(float)/1e9
    mean_uniqness = __sampleUniqness(idx, itv)
    mean_uniqness = (time_interval).join(
        pd.DataFrame(mean_uniqness, columns=["uniqness"])
    )
    return mean_uniqness


@numba.jit(nopython=True)
def __numBarInRet(time_index, time_interval):
    # 各リターンがbarをいくつ含むかを計算(mat.sum(axis=0))
    count_arr = np.zeros(time_interval.shape[0])
    _i = 0
    I = len(time_index)
    for j in range(len(time_interval)):  # return sample loop:j
        for i in range(_i, I):  # bar loop:i
            if time_index[i] < time_interval[j, 0]:
                _i = i
            elif time_index[i] <= time_interval[j, 1]:
                count_arr[j] += 1
            elif time_interval[j, 1] < time_index[i]:
                break
    return count_arr


@numba.jit(nopython=True)
def __c_t(time_index, time_interval):
    # 各barがリターンに何回現れるかを計算(mat.sum(axis=1))
    # __sampleUniqnessで使用.
    count_arr = np.zeros(time_index.shape[0])
    _i = 0
    I = len(time_index)
    for j in range(len(time_interval)):  # return sample loop:j
        for i in range(_i, I):  # bar loop:i
            if time_index[i] < time_interval[j, 0]:
                _i = i
            elif time_index[i] <= time_interval[j, 1]:
                count_arr[i] += 1
            elif time_interval[j, 1] < time_index[i]:
                break
    return count_arr


@numba.jit(nopython=True)
def __sampleUniqness(time_index, time_interval):
    # リターンiの平均独自性を計算
    uniqness = np.zeros(time_interval.shape[0])
    c_t = __c_t(time_index, time_interval)  # 各barのリターン中での出現回数:index=i
    n_b = __numBarInRet(time_index, time_interval)  # 各リターン中でのbarの出現回数:index=j
    _i = 0
    I = len(time_index)
    for j in range(len(time_interval)):  # return sample loop
        for i in range(_i, I):  # bar loop
            if time_index[i] < time_interval[j, 0]:
                _i = i
            elif time_index[i] <= time_interval[j, 1]:
                uniqness[j] += 1 / c_t[i]
            elif time_interval[j, 1] < time_index[i]:
                break
    return uniqness / n_b


def getIntegratedReturn(events, midPrice, theta_ver):
    """ eventからtheta_verまでの時間で積分されたリターンを返す.
    Args:
        events (Series or DatetimeIndex): sampled event timestamps.
        midPrice (pd.DF or Series with DatetimeIndex):
        theta_ver (timedelta): threshold of vertical barrier
    """
    assert "M8" in events.dtype.str, "events must be DatetimeIndex."
    assert "M8" in midPrice.index.dtype.str, "midPrice must have DatetimeIndex."
    _midPrice = midPrice.sort_index().reset_index().values
    _midPrice[:, 0] = midPrice.index.values.astype(float)/1e9
    _midPrice = _midPrice.astype(float)  # [N,2]
    _events = events.values.astype(float)/1e9  # [n,]
    ret = __integratedReturn(_events, _midPrice, theta_ver.total_seconds())
    result = pd.DataFrame({
        "t0": pd.to_datetime((_events*1e9)).round("ms").values,
        "t1": events+theta_ver,
        "ret": ret
    })
    return result


@numba.jit(nopython=True)
def __integratedReturn(events, midPrice, theta_ver):
    # 時間で積分したリターンを返す.
    _t = 0
    _price_init = -1  # initial price of sample
    _price_last = -1
    _time_last = -1  # unixtime
    T = max(midPrice[:, 0])
    ret = np.zeros(events.shape[0])
    for i in range(len(ret)):  # return sample loop:i
        for t in range(_t, len(midPrice)):  # time sample loop:t
            if T < events[i]+theta_ver:
                ret[i] = np.nan  # リターンが計算可能な時間外の場合nan
                break
            elif midPrice[t, 0] < events[i]:
                pass
            elif events[i] <= midPrice[t, 0] and midPrice[t, 0] < events[i]+theta_ver:
                if _price_init < 0:
                    _time_last = midPrice[t, 0]
                    _price_init = _price_last = midPrice[t, 1]
                    _t = max(t-1, 0)
                ret[i] += (_price_last - _price_init) * \
                    (midPrice[t, 0] - _time_last)
                _time_last = midPrice[t, 0]
                _price_last = midPrice[t, 1]
            else:
                _time_last = -1
                _price_init = -1
                _price_last = -1
                break
    return ret



def getFirstSignal(signal_df, horizon):
    """ horizon秒を期限に重複するsignalを除く
    Args:
        signal_df (pd.DataFrame): [label, t0, t1]のdf. labelはU,N,D.
        horizon (float): signalの賞味期限[sec].
    Returns:
        uniq_signal (pd.DataFrame): 
    """
    assert isinstance(signal_df, pd.DataFrame)
    label_col = signal_df.columns[signal_df.columns.str.contains("label")]
    assert label_col.shape[0]==1, "signal_df dataframe have multiple or no label in column."

    label_name = label_col[0]
    _label = np.argmax(get_und_dummies(signal_df[label_name]).values, axis=1) - 1
    _t0 = signal_df.t0.values.astype(float) /1e9
    _t1 = signal_df.t1.values.astype(float) /1e9

    _label, _t0, _t1, n_duplicates = __getFirstSignal(_label, _t0, _t1, horizon)

    _t0 = pd.to_datetime(_t0*1e9).round("ms")
    _t1 = pd.to_datetime(_t1*1e9).round("ms")
    uniq_signal = pd.DataFrame(
        {"t0": _t0, "t1": _t1, label_name: _label, "duplicates": n_duplicates}
    )
    uniq_signal[label_name] = uniq_signal[f"{label_name}"].replace(-1,"D").replace(+1,"U").replace(0,"N")
    return uniq_signal


@numba.jit(nopython=True)
def __getUniqueSignal(label, t0, t1):
    # label:-1,0,1のndarray, t0,t1: timestamp, extend: bool
    n_duplicates = np.zeros(label.shape[0])
    I = len(label)
    for i in range(I):  # 全signal iについて
        
        if label[i] == 0:
            # Nsignalならpass
            continue
            
        for j in range(i+1, I):  # 後に出た signal j (> i)について
            if t1[i] < t0[j]:
                # 後のsignalがhorizonの外で発生 => 当初のt1でOK
                t1[i] = t1[i]
                break
            else:
                # 後のsignalがhorizonの内側で発生
                if label[j] == 0:
                    # 後のsignalがNsignal
                    pass
                elif label[i]*label[j] > 0:
                    # 後のsignalが同一方向
                    # => 後のsignalをNラベルに書き換える
                    # => (extendの場合 現在のsignalの有効期限を後のsignalのものに引き延ばす)
                    label[j] = 0
                    t1[i] = t1[j]
                    n_duplicates[i] += 1  # 重複数を確認しておく
                elif label[i]*label[j] < 0:
                    # 後に発生したsignalが逆側 => ここで打ち止め
                    t1[i] = t0[j]
                    break

    return label, t0, t1, n_duplicates


def getUniqueSignal(signal_df):
    """ 重複するsignalを除き, 一番最初のsignalのみ残す.
    Args:
        signal_df (pd.DataFrame): [label, t0, t1]のdf. labelはU,N,D.
        expiration_seconds (bool): signalの賞味期限[sec].
    Returns:
        uniq_signal (pd.DataFrame): 
    """
    assert isinstance(signal_df, pd.DataFrame)
    label_col = signal_df.columns[signal_df.columns.str.contains("label")]
    assert label_col.shape[0]==1, "signal_df dataframe have multiple or no label in column."

    label_name = label_col[0]
    _label = np.argmax(get_und_dummies(signal_df[label_name]).values, axis=1) - 1
    _t0 = signal_df.t0.values.astype(float) /1e9
    _t1 = signal_df.t1.values.astype(float) /1e9

    _label, _t0, _t1, n_duplicates = __getUniqueSignal(_label, _t0, _t1)

    _t0 = pd.to_datetime(_t0*1e9).round("ms")
    _t1 = pd.to_datetime(_t1*1e9).round("ms")
    uniq_signal = pd.DataFrame(
        {"t0": _t0, "t1": _t1, label_name: _label, "duplicates": n_duplicates}
    )
    uniq_signal[label_name] = uniq_signal[f"{label_name}"].replace(-1,"D").replace(+1,"U").replace(0,"N")
    return uniq_signal


def to3classLabel(array_1d, threshold=[0, 0], isPercentage=False):
    """ 1d-float -> 1d-UND
    Args:
        array_1d (pd.DataFrame): 予測値もしくはリターンの1d-array.
        threshold (list of float): 
    Returns:
        (pd.DataFrame): 
    """
    assert hasattr(array_1d, "index"), "array_1d must be pd.Series or DataFrame."

    if isPercentage:
        if threshold[1]<1: warnings.warn(f"percentile threshold should be set within [0,100], not [0,1]")
        _threshold = np.percentile(array_1d, q=threshold)
        arr_labeled = array_1d\
            .mask(array_1d<=_threshold[0], "D")\
            .mask(array_1d>=_threshold[1], "U")\
            .mask((_threshold[0]<array_1d)&(array_1d<_threshold[1]), "N")
        return arr_labeled

    else:
        arr_labeled = array_1d\
            .mask(array_1d<=threshold[0], "D")\
            .mask(array_1d>=threshold[1], "U")\
            .mask((threshold[0]<array_1d)&(array_1d<threshold[1]), "N")
        return arr_labeled


def fromPredToSignal(y_pred, threshold=None, isPercentage=False):
    """ 3d-float -> 1d-UND
    Args:
        y_pred (np.ndarray): 3クラス分類器のoutput. shapeは[nsample, 3]で各要素は0<=elem<=1の値.
        thredshold (float): signal発生閾値
        isPercentileThreshold (bool): thresholdをpercentile(0<=.<=1)で渡すかどうか
    Returns:
        (np.ndarray): [0,1,0,1,2...]
    """
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape[1]==3

    if isPercentage:
        if threshold<1: warnings.warn(f"percentile threshold should be set within [0,100], not [0,1]")
        prob = np.append(y_pred[:,0], y_pred[:,2])
        val_threshold = np.percentile(prob, threshold)
        return fromPredToSignal(y_pred, val_threshold, isPercentage=False)

    else:
        if threshold is None:
            # 閾値なしの場合は最大のclassを出力する
            signal = np.argmax(y_pred, axis=1).astype(object)
            signal[signal==0] = "D"
            signal[signal==1] = "N"
            signal[signal==2] = "U"
            return signal
        else:
            # (3値分類のみ正常) 閾値ありの場合,閾値を超えたクラスのうち最大のものをPositiveとする
            signal_D = ((y_pred[:,0] > threshold) & (y_pred[:,0] > y_pred[:,2])).reshape(-1, 1)
            signal_U = ((y_pred[:,2] > threshold) & (y_pred[:,2] > y_pred[:,0])).reshape(-1, 1)
            signal_N = ((y_pred[:,0] <= threshold) & (y_pred[:,2] <= threshold)).reshape(-1, 1)
            signal = 1 * np.concatenate([signal_D, signal_N, signal_U], axis=1)
            signal = np.argmax(signal, axis=1).astype(object)
            signal[signal==0] = "D"
            signal[signal==1] = "N"
            signal[signal==2] = "U"
            return signal