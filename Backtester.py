import numpy as np
import pandas as pd
import numba
from typing import Callable,Dict,List
import warnings
warnings.filterwarnings("ignore")



class Backtester:
    """ 1sの簡易バックテスト.
        - backetest()にStratetgyクラスのメソッド:actionを渡してテストを行う.
        - action(t, state, feature)ではstateの書き換えを行う.
        - stateに書き込まれたorder,position情報を1secごとに更新&記録.
        - stateの履歴を返却.
    """
    def __init__(self, book:pd.DataFrame, activity:pd.DataFrame, latency=0):
        act = activity.set_index("mktTime")
        self._book = book
        self._act = act
        self.latency = latency #[msec]
        self.setPrice()

    
    def setPrice(self):
        """ latencyを加味してPriceを作成 """ 
        act = self._act
        book = self._book.shift(self.latency, freq='ms')
        # LO約定用Price: activityのプライスから1秒ごと(min,max)にresample
        limitAsk = act.query("side=='buy'").price.resample("1s").max().to_frame(name="limitAsk")
        limitBid = act.query("side=='sell'").price.resample("1s").min().to_frame(name="limitBid")
        # MO約定用Price: bookのPriceからSecondを1秒ごとにresample(last)
        bestAsk = book.loc[:,"askPrice1"].resample("1s").last().to_frame(name="bestAsk").fillna(method="ffill").dropna()
        bestBid = book.loc[:,"bidPrice1"].resample("1s").last().to_frame(name="bestBid").fillna(method="ffill").dropna()
        # 累積板厚 :投資行動からみて直近の秒足closeの値
        bidVol = book.loc[:,book.columns.str.contains("bidVol")].resample("1s").last().shift().cumsum(axis=1).fillna(method="ffill").fillna(method="ffill",axis=1)
        askVol = book.loc[:,book.columns.str.contains("askVol")].resample("1s").last().shift().cumsum(axis=1).fillna(method="ffill").fillna(method="ffill",axis=1)
        # 各レベルの価格
        bidPrices = book.loc[:,book.columns.str.contains("bidPrice")].resample("1s").last().shift().fillna(method="ffill").fillna(method="ffill",axis=1)
        askPrices = book.loc[:,book.columns.str.contains("askPrice")].resample("1s").last().shift().fillna(method="ffill").fillna(method="ffill",axis=1)
        # indexの積集合を作り,そこにjoinすることで配列サイズを統一する
        idx = (limitBid.index & limitAsk.index & bestBid.index & bestAsk.index & bidVol.index & askVol.index)
        idx = pd.DataFrame(index=idx); self.idx = idx
        print(idx.index[0], " ~ ", idx.index[-1])
        # attr作成
        self._bestAsk=idx.join(bestAsk); self._limitAsk=idx.join(limitAsk)
        self._bestBid=idx.join(bestBid); self._limitBid=idx.join(limitBid)
        self._midPrice = pd.DataFrame((self._bestBid.values+self._bestAsk.values)/2, index=idx.index, columns=["midPrice"])
        self._bidVol = idx.join(bidVol); self._bidPrices = idx.join(bidPrices)
        self._askVol = idx.join(askVol); self._askPrices = idx.join(askPrices)
        self.T = self._midPrice.shape[0]
        self._feature = idx.copy()


    def addFeature(self, data:pd.DataFrame, fillna=0):
        """ actionに必要なデータを渡す.
            data (pd.DF with DatetimeIndex):
        """
        assert "M8" in data.index.dtype.str, "input data must have DatetimeIndex"
        self._data = data
        self._feature = pd.merge_asof(self._feature, self._data.shift(self.latency,freq='ms'), 
                                      tolerance=pd.Timedelta('1s'),
                                      left_index=True, right_index=True)
        if fillna == "ffill":
            self._feature.fillna(method="ffill", inplace=True)
        else:
            self._feature.fillna(fillna, inplace=True)


    def resetLatecy(self, latency):
        self.latency = latency
        self.setPrice()
        if hasattr(self, "_data"): self.addFeature(self._data)


    @property
    def prices(self):
        return [self._bestBid.values.reshape(-1,), self._bestAsk.values.reshape(-1,),
                self._limitBid.values.reshape(-1,), self._limitAsk.values.reshape(-1,)]


    def backtest(self, action:Callable, q_limit=0.01)-> List[np.array]:

        # Playするマーケットの情報
        bestBid, bestAsk, limitBid, limitAsk = self.prices
        bidVol, askVol = self._bidVol.values, self._askVol.values
        bidPrices, askPrices = self._bidPrices.values, self._askPrices.values
        midPrice = self._midPrice.values.reshape(-1,)

        # transaction時系列[price,amount]@time t
        Q = np.zeros(self.T)
        trs_bid = np.zeros([self.T,2]); trs_buy  = np.zeros([self.T,2])
        trs_ask = np.zeros([self.T,2]); trs_sell = np.zeros([self.T,2])
        pnl_close = np.zeros(self.T); pnl_open = np.zeros(self.T)
        api_count = np.zeros(self.T)

        # 情報@t
        feature = np.array(list(self._feature.reset_index().to_dict("index").values()))

        # 時刻tでの状態 :
        state = {
            "Q" : 0.,
            "p_deal" : -1.,
            "pnl_open"  : 0.,
            "pnl_close" : 0.,
            "api_count" : 0.,
            "bid" : [], #LOs: [np.array([t,price,amount]),...]
            "ask" : [],
            "buy" : [0,0.], #MOs: [time, amount]
            "sel" : [0,0.],
            "q_limit" : q_limit
        }

        # バックテスト実行
        for t in np.arange(self.T):
            if t%86400 == 0: print(t)

            # 投資行動の決定 action:state -> order書換
            state["api_count"] = max(0, state["api_count"]-1)
            action(t, state, feature[t], bidPrices[t], bidVol[t], askPrices[t], askVol[t])

            # check約定 : LO
            if state["bid"] and not np.isnan(limitBid[t]):
                trs_bid[t], state["bid"] = Backtester._checkLO(t, state["bid"], limitBid[t], +1)
            if state["ask"] and not np.isnan(limitAsk[t]):
                trs_ask[t], state["ask"] = Backtester._checkLO(t, state["ask"], limitAsk[t], -1)

            # check約定 : MO
            if 0<state["buy"][0] and state["buy"][0] <= t: #buy
                trs_buy[t,0] = bestAsk[t]      #価格
                trs_buy[t,1] = state["buy"][1] #数量
                state["buy"][0] = 0; state["buy"][1] = 0

            if 0<state["sel"][0] and state["sel"][0] <= t: #sell
                trs_sell[t,0] = bestBid[t]      #価格
                trs_sell[t,1] = state["sel"][1] #数量
                state["sel"][0] = 0; state["sel"][1] = 0

            ## オープン/クローズ/ポジション評価 ##
            state["Q"], state["p_deal"], state["pnl_open"], current_pnl_close =\
            Backtester.position_evaluate(
                state["Q"], state["p_deal"], midPrice[t],
                trs_bid[t], trs_ask[t], trs_buy[t], trs_sell[t]
            )
            state["pnl_close"] += current_pnl_close
            if state["Q"]!=Q[t-1]: state["api_count"]+=1
            Q[t] = state["Q"]
            pnl_open[t]  = state["pnl_open"]
            pnl_close[t] = state["pnl_close"]
            api_count[t] = state["api_count"]
            ###

        print("complete.")
        return [trs_bid, trs_ask, trs_buy, trs_sell, Q, pnl_close, pnl_open, api_count]


    @staticmethod
    @numba.jit(nopython=True)
    def position_evaluate(Q:float, p_deal:float, midPrice:float,\
                          trs_bid:np.array, trs_ask:np.array, trs_buy:np.array, trs_sell:np.array):
        ## オープン/クローズ/ポジション評価 ##

        newQ = round(Q+trs_bid[1]-trs_ask[1]+trs_buy[1]-trs_sell[1], 3)
        if newQ * Q < 0:
            # ポジション転換 : クローズ処理+新たな建玉の評価
            execPrice = (trs_bid.prod() - trs_ask.prod() + trs_buy.prod() - trs_sell.prod())\
                        / (trs_bid[1] - trs_ask[1] + trs_buy[1] - trs_sell[1])
            pnl_close = (execPrice - p_deal) * Q
            p_deal = execPrice

        elif newQ * Q > 0:
            # ポジション保持 : 現在の建玉の修正と記録
            p_deal = (
                (Q * p_deal)
                + (trs_bid.prod() - trs_ask.prod())
                + (trs_buy.prod() - trs_sell.prod())
            ) / ( newQ )
            # ポジション損益の評価@midPrice
            pnl_open = (midPrice - p_deal)*newQ

        else:
            if Q != 0 and newQ == 0:
                # クローズ
                execPrice = (trs_bid.prod() - trs_ask.prod() + trs_buy.prod() - trs_sell.prod())\
                            / (trs_bid[1] - trs_ask[1] + trs_buy[1] - trs_sell[1])
                pnl_close = (execPrice - p_deal) * Q
                p_deal = -1
            elif Q == 0 and newQ != 0:
                # エントリ
                p_deal = (trs_bid.prod() - trs_ask.prod() + trs_buy.prod() - trs_sell.prod())\
                            / (trs_bid[1] - trs_ask[1] + trs_buy[1] - trs_sell[1])

        return [newQ, p_deal, pnl_open, pnl_close]


    @staticmethod
    @numba.jit(nopython=True)
    def _checkLO(t:int, orders:List, limitPrice:float, bidFlag:int):
        """
            orders  : [np.array([time,price,amount]), np.array(),...]
            bidFlag : +1(bid), -1(ask)
        """
        trs = []
        remain_orders = []
        for i in range(len(orders)):
            if orders[i][0] <= t:
                if   bidFlag > 0 and limitPrice <= orders[i][1]:
                    trs.append(orders[i][1:])
                elif bidFlag < 0 and orders[i][1] <= limitPrice:
                    trs.append(orders[i][1:])
            else:
                remain_orders.append(orders[i])

        if len(trs)>1:
            # 2個以上の注文が約定した場合1つにまとめる
            totalamount = 0.
            weighted_price = 0.
            for i in range(len(trs)):
                weighted_price += trs[i][0] * trs[i][1]
                totalamount += trs[i][1]
            trs.clear()
            trs.append(np.array([weighted_price/totalamount, totalamount]))
        elif len(trs)==0:
            trs.append(np.array([0., 0.]))

        return trs[0], remain_orders #np.array([price,amount]), [np.array]