import numpy as np
import pandas as pd
import numba
import os


def getTickInbbar(market_activity, initial_T=10):
    """ 約定価格の価格変化の方向の偏りからbarを作成
    Args:
        market_activity (pd.DF):
        initial_T (int): 典型的なbarの期間Tの初期値
    """
    assert "M8" in market_activity.index.dtype.str, \
           "market_activity must have DatetimeIndex"
    assert type(market_activity) == pd.DataFrame, \
           "market_activity must be pandas.DataFrame"

    dP = market_activity.price.diff().to_frame("dP").dropna()
    dP["flag"] = (
        1*(dP > 0) - 1*(dP < 0) + (dP == 0).replace(True, np.nan)
    ).fillna(method="ffill")  # 同一の価格の場合には前回の符号を参照
    prob_positive = dP.flag.ewm(alpha=0.5).mean().values
    prob_positive[0] = 0.5

    bar, theta, expected_T = [0], 0, initial_T
    for t, b, prob in zip(np.arange(dP.shape[0]), dP.flag.values, prob_positive):
        theta += b
        if np.abs(theta) > expected_T * np.abs(2*prob-1):
            theta = 0
            expected_T = expected_T*0.5 + (t-bar[-1])*0.5
            bar.append(t)
    return dP.index[bar]


def getVolbar(market_activity, threshold, isBase=True):
    """ ボリュームバーもしくはドルバーを作成.(BASE/TERM)
        * volume bar...約定量が一定量を超えた場合にサンプリング.偏りは考えない.
    """
    if isBase:
        cumAmount = market_activity.amount.cumsum()
    else:
        cumAmount = (market_activity.price*market_activity.amount).cumsum()

    flag = np.floor(cumAmount/threshold)
    flag = flag.diff().to_frame("flag")
    bar = flag.query("flag>0").index
    return bar


def cusumFilter(raw_series, theta, mean_func="exp", window=0.5):
    """ 対称CUSUM Filter. 系列St(raw_series)の平均乖離の累積値がtheta以上になったらsignal.
    Args:
        raw_series (pd.Series indexed by time): 部分定常時系列. (e.g. price)
        theta (numeric): イベント発生の閾値
    """
    assert "M8" in raw_series.index.dtype.str,\
           "raw_series must have DatetimeIndex"
    if mean_func is None: dS = raw_series.diff().fillna(0)
    elif mean_func == 'exp': dS = (raw_series - raw_series.ewm(alpha=window).mean()).dropna()
    elif mean_func == 'MA': dS = (raw_series - raw_series.rolling(seconds=window).mean()).dropna()
    events_p, events_n = __cusum_filter(dS.values.astype(float), float(theta))
    return raw_series.index[events_p], raw_series.index[events_n]


@numba.jit(nopython=True)
def __cusum_filter(dS: np.array, theta: float):
    events_p = []
    events_n = []
    Sp = 0.
    Sn = 0.
    for i in range(len(dS)):
        Sp, Sn = max(0, Sp+dS[i]), min(0, Sn+dS[i])
        if Sn < -theta:
            Sn = 0; events_n.append(i+1)
        elif Sp > +theta:
            Sp = 0; events_p.append(i+1)
    return [events_p, events_n]


def getFisherSignal(series, eff_pval=1, window=60):
    """　Fisher検定によるトレンド検出シグナル
    Args:
        series (pd.Series): DatetinmeIndexつき原系列
        eff_pval (float, optional): 実効P値[%]. Defaults to 1.
        window (int, optional): lookbackの長さ. Defaults to 60.

    Returns:
        pd.DataFrame: [変化点時刻,検出時刻,方向のdataframe]
    """

    df_effPval = pd.read_csv(os.path.dirname(__file__)+"/fisher_effPval.csv")
    assert eff_pval in df_effPval.q.unique() and window in df_effPval.W.unique(),\
        "eff_pval or window is invalid these values must be included in fisher_effPval.csv"

    time_change, time_signal, direction =\
        _fisher_exact_test_bulk(
            series.values,
            df_effPval.query("W==@window and q==@eff_pval").theta_p.iloc[0],
            window
        )
    data = pd.DataFrame({
        "change_point": series.index[time_change[direction != 0].astype(int)],
        "signal": series.index[time_signal[direction != 0].astype(int)],
        "direction": direction[direction != 0]
    })
    return data


@numba.jit(nopython=True)
def _fisher_exact_test_bulk(series, pval, window):
    """ 入力した時系列全てに渡ってfisher検定を行う
    """
    time_change = np.zeros_like(series)
    time_signal = np.zeros_like(series)
    direction = np.zeros_like(series)
    for t in range(window, series.shape[0]):
        _lb, _direction, _pval = _fisher_exact_test_main(series[t-window:t])
        if _pval < pval:
            time_change[t] = t - _lb
            time_signal[t] = t
            direction[t] = _direction
    return time_change, time_signal, direction


logsumVec = np.array(
    [0] + [np.log(np.arange(1, k+1)).sum() for k in np.arange(1, 1000)])


@numba.jit(nopython=True)
def _fisher_exact_test_main(series: np.array):
    """ fisher検定のmain処理
    """
    q_values = np.quantile(series, q=[0.25, 0.5, 0.75]) + 1e-6
    W = series.shape[0]
    w = int(W/2+1)
    p_values = 100*np.ones(w*3)
    directions = np.zeros(w*3)
    for i, q_val in enumerate(q_values):
        for div_lookback in range(1, W/2+1):
            # count up each quadrant's points
            t_div = W - (div_lookback)
            a, b, c, d = 0, 0, 0, 0
            for t, val in enumerate(series):
                if val > q_val and t < t_div: a += 1  # 1象限
                elif val < q_val and t < t_div: c += 1  # 3
                elif val > q_val and t_div <= t: b += 1  # 2
                elif val < q_val and t_div <= t: d += 1  # 4

            # calculate fisher p-values
            n = a + b + c + d
            p_value = 0
            if a*d - b*c > 0:
                # P(> a,b,c,d)を返す.Sumupする.
                threshold = c if b > c else b
                m = a + threshold
                directions[div_lookback + w*i] = -1
                for k in range(a, m+1):
                    p_value += np.exp(
                        logsumVec[a+b] + logsumVec[c+d] + logsumVec[a+c]
                        + logsumVec[b+d] - logsumVec[k] - logsumVec[a+b-k]
                        - logsumVec[a+c-k] - logsumVec[d-a+k] - logsumVec[n]
                    )
            else:
                # P(< a,b,c,d)を返す.Sumdownする.
                threshold = d if a > d else a
                m = a - threshold
                directions[div_lookback + w*i] = +1
                for k in range(m, a+1):
                    p_value += np.exp(
                        logsumVec[a+b] + logsumVec[c+d] + logsumVec[a+c]
                        + logsumVec[b+d] - logsumVec[k] - logsumVec[a+b-k]
                        - logsumVec[a+c-k] - logsumVec[d-a+k] - logsumVec[n]
                    )
            p_values[div_lookback + w*i] = p_value

    t = np.argmin(p_values)
    return t % w, directions[t], np.min(p_values)


@numba.jit(nopython=True)
def _get_first_signal(signal_time, lookforward):
    flags = np.zeros_like(signal_time)
    last_signal = 0
    for i, signal in enumerate(signal_time):
        if last_signal + lookforward < signal:
            flags[i] = 1
            last_signal = signal
    return flags


def getFirstSignal(signal_time, lookforward):
    """ 予測ホライズン内で重複するシグナルを削除する
    Args:
        signal_time (DatetimeIndex): シグナルイベント時刻
        lookforward (type): 予測ホライズン[sec]
    Returns:
        DatetimeIndex: 重複が除かれたシグナル
    """
    flags = _get_first_signal(
        signal_time.values.astype(float)/1e9, lookforward
    )
    return signal_time[flags.astype(bool)]
