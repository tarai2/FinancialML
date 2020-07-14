module Backtester

using DataFrames
using Dates
using DataStructures
using PyCall
pd = pyimport("pandas")

export MarketActivity, MarketBook, Item, MarketData, Order, Position,
        make_book, make_activity, backtest, push_api!, count_api


struct MarketActivity
    isBuy::Bool
    price::Float32
    amount::Float32
end

struct MarketBook
    bidPrices::Vector{Float32}
    askPrices::Vector{Float32}
    bidVols::Vector{Float32}
    askVols::Vector{Float32}
end

struct DiffMarketBook
    bids::Vector{Float32}
    asks::Vector{Float32}
end

struct Item
    market::String
    channel::String #"deal", "activity", "book", "diffbook"
    data::Union{Vector{MarketActivity}, MarketBook, DiffMarketBook}

    Item(market::String, channel::String, act::MarketActivity) = new(market, channel, [act])
    Item(market::String, channel::String, book::MarketBook) = new(market, channel, book)
end

mutable struct Position
    price::Float32
    amount::Float32
    pnl_open::Float32
    pnl_close::Float32

    Position() = new(NaN,0,0,0)
    Position(position::Position) = new(position.price, position.amount, position.pnl_open, position.pnl_close)
end

mutable struct Order
    time::Float64
    type::String #"Bid","Ask","Buy","Sell"
    price::Float32
    amount::Float32
end


### 約定判定
function dealing(time::Float64, orders::Vector{Order}, activity::Vector{MarketActivity})::
    Vector{Vector{Order}}
    """ LimitOrderの約定確認.
        ボリュームは加味せず常に全量約定. ボリューム加味...各orderにつきsortされたactivityを舐める.
    """
    if length(orders)>0
        buy_prices = [act.price for act in activity if  act.isBuy]
        sel_prices = [act.price for act in activity if !act.isBuy]
        max_buy = length(buy_prices)>0 ? maximum(arr) : +Inf
        min_sel = length(sel_prices)>0 ? minimum(arr) : -Inf

        dealt_orders = Order[]
        remain_orders = Order[]
        for order in orders
            if order.time < time
                if order.type == "Ask"
                    if order.price <= max_buy
                        push!(dealt_orders, order)
                    else
                        push!(remain_orders, order)
                    end
                elseif order.type == "Bid"
                    if min_sel <= order.price
                        push!(dealt_orders, order)
                    else
                        push!(remain_orders, order)
                    end
                end
            else
                push!(remain_orders, order)
            end
        end
        return [remain_orders, dealt_orders]
    else
        return [orders, orders]
    end
end


function dealingV(time::Float64, orders::Vector{Order}, activity::Vector{MarketActivity})::
    Vector{Vector{Order}}
    """ LO約定. Volume加味.
    """
    MO_buy = sort([[act.price, act.amount] for act in activity if  act.isBuy], rev=false)#昇順
    MO_sel = sort([[act.price, act.amount] for act in activity if !act.isBuy], rev=true) #降順
    dealt_orders = Order[]
    remain_orders = Order[]
    # 各Orderを約定確認
    if order.time < time
        for order in sort([o for o in orders if o.type=="Bid"], by=o->o.price, rev=true)
            for buy in MO_buy
                if buy[1] < order.price
                    # 価格足らず
                    # pass
                else
                    if buy[2] <= 0.0
                        # 処理済みのMO
                        # pass
                    elseif buy[2] <= order.amount
                        # 部分約定
                        push!(dealt_orders, Order(time, order.type,order.price, buy[2]))
                        order.amount -= buy[2]
                        push!(remain_orders, order)
                        buy[2] = 0.0
                    elseif buy[2] > order.amount
                        # 全約定
                        push!(dealt_orders, order)
                        buy[2] -= order.amount
                    end
                end
            end
        end
        for order in sort([o for o in orders if o.type=="Ask"], by=o->o.price, rev=false)
            for sel in MO_sel
                if sel[1] > order.price
                    # 価格足らず
                    # pass
                else
                    if sel[2] <= 0.0
                        # 処理済みMO
                        # pass
                    elseif sel[2] <= order.amount
                        # 部分約定
                        push!(dealt_orders, Order(time, order.type,order.price, sel[2]))
                        order.amount -= sel[2]
                        push!(remain_orders, order)
                        sel[2] = 0.0
                    elseif sel[2] > order.amount
                        # 全約定
                        push!(dealt_orders, order)
                        sel[2] -= order.amount
                    end
                end
            end
        end

    else
        # 取引所に未到着の注文
        push!(remain_orders, order)
    end
    return [remain_orders, dealt_orders]
end


function dealing(time::Float64, orders::Vector{Order}, book::MarketBook)::
    Vector{Vector{Order}}
    """ MOの約定確認. ボリュームは加味せず常にベスト約定.
    """
    dealt_orders  = filter(odr->odr.time<=time, orders)
    remain_orders = filter(odr->odr.time >time, orders)
    for order in dealt_orders
        if order.type == "Buy"
            order.price = book.askPrices[1]
        elseif order.type == "Sell"
            order.price = book.bidPrices[1]
        end
    end
    return [remain_orders, dealt_orders]
end


### ポジション更新
function update_position(orders::Vector{Order}, position::Position, midPrice::Float64)::Position
    """ ポジション情報を更新
    """
    dQ = 0.0
    dQ_eval = 0.0
    for order in orders
        if order.type == "Bid" || order.type == "Buy"
            dQ += order.amount
            dQ_eval += order.amount*order.price
        elseif order.type == "Ask" || order.type == "Sell"
            dQ -= order.amount
            dQ_eval -= order.amount*order.price
        end
    end

    new_position = Position(position)
    if dQ != 0.0
        newQ = round(position.amount + dQ, digits=4)
        if position.amount * newQ < 0
            # ポジション転換 : クローズ+新たな建玉の評価
            execPrice = dQ_eval / dQ
            new_position.pnl_close += (execPrice - position.price) * position.amount
            new_position.price  = execPrice
            new_position.amount = newQ
            new_position.pnl_open = (midPrice - execPrice) * newQ

        elseif position.amount * newQ > 0
            # ポジション維持
            dealPrice = (position.price*position.amount + dQ_eval) / newQ
            new_position.price = dealPrice
            new_position.amount = newQ
            new_position.pnl_open = (midPrice - position.price) * position.amount
        else
            if (position.amount!=0) & (newQ==0)
                # クローズ
                dealPrice = dQ_eval / dQ
                new_postion.pnl_close += (position.price - dealPrice)*position.amount
                new_position.price  = NaN
                new_position.amount = 0.
                new_position.pnl_open = 0.
            elseif (position.amount==0.) & (newQ!=0.)
                # エントリ
                new_position.price = dQ_eval / dQ
                new_postiion.amount = newQ
                new_position.pnl_open = (midPrice - position.price) * position.amount
            end
        end
        return new_position
    else
        return new_position
    end
end


### データ整形
function pandas2jldf(df::PyObject)::DataFrame
    # 各列ごとにDataFramesに変換, その後concatする
    d = DataFrame[]
    for (k,v) in df.iteritems()
        if v.dtype == "<M8[ns]" #Datetime
            coldata = DataFrame(reshape([unix2datetime(t) for t in v.values.astype("float64")./1e9], length(df), 1))
            rename!(coldata, [Symbol(k)])
        elseif v.dtype == "float64" #Float
            coldata = DataFrame(reshape(v.values, length(df), 1))
            rename!(coldata, [Symbol(k)])
        else #String <- Seriesの上でiterationを回すのでこいつで時間がかかる
            coldata = DataFrame(reshape([s for s in v], length(df),1))
            rename!(coldata, [Symbol(k)])
        end
        push!(d, coldata)
    end
    return hcat(d...)
end

function make_activity(market::String, df::PyObject, which_time="mktTime")::
    SortedDict{Float64, Item}

    channel = "activity"
    res = SortedDict{Float64, Item}()
    newdf = pandas2jldf(df)
    if which_time == "mktTime"
        rename!(newdf, :mktTime=>Symbol("time"))
        select!(newdf, Not(:rcvTime))
        channel = "deal"
    elseif which_time == "rcvTime"
        rename!(newdf, :rcvTime=>Symbol("time"))
        select!(newdf, Not(:mktTime))
        channel = "activity"
    end

    for row in eachrow(newdf)
        if Dates.datetime2unix(row.time) in keys(res)
            # 追加
            push!(
                res[Dates.datetime2unix(row.time)].data,
                MarketActivity(
                    ifelse(row.side=="buy",true,false), row.price,　row.amount)
            )
        else
            # 新規作成
            res[Dates.datetime2unix(row.time)] =
                Item(
                    market,
                    channel,
                    MarketActivity(
                        ifelse(row.side=="buy",true,false),
                        row.price,
                        row.amount)
                )
        end
    end
    res
end

function make_book(market::String, df::PyObject ; level::Int64=10, ceiling::Int64=0)::
    SortedDict{Float64, Item}

    channel = "book"
    res = SortedDict{Float64, Item}()
    # Priceを取り出しDataFramesに整形
    newdf = pandas2jldf(df)
    if ceiling > 0
        newdf[:, :rcvTime] = map( x->Dates.ceil(x,Dates.Millisecond(ceiling)), newdf[:, :rcvTime] );
        newdf = DataFrame(vcat([last(x) for x in (groupby(newdf, :rcvTime))]));
    end
    col_bP = sort( filter(x->occursin("bidPrice",String(x)),names(newdf)),
                   by=x->parse(Int32, String(match(r"([0-9])+",String(x)).match)) )
    col_bV = sort( filter(x->occursin("bidVol",String(x)),names(newdf)),
                   by=x->parse(Int32, String(match(r"([0-9])+",String(x)).match)) )
    col_aP = sort( filter(x->occursin("askPrice",String(x)),names(newdf)),
                   by=x->parse(Int32, String(match(r"([0-9])+",String(x)).match)) )
    col_aV = sort( filter(x->occursin("askVol",String(x)),names(newdf)),
                   by=x->parse(Int32, String(match(r"([0-9])+",String(x)).match)) )
    bidPrices = view(newdf, :, col_bP[1:level])
    askPrices = view(newdf, :, col_aP[1:level])
    bidVols = view(newdf, :, col_bV[1:level])
    askVols = view(newdf, :, col_aV[1:level])
    rcvTime = view(newdf, :, :rcvTime)

    for (time,bidPrice,askPrice,bidVol,askVol) in zip(rcvTime, eachrow(bidPrices), eachrow(askPrices),
                                                               eachrow(bidVols),   eachrow(askVols))
        # 新規作成
        res[Dates.datetime2unix(time)] =
            Item(
                market,
                channel,
                MarketBook(
                    Vector{Float32}(bidPrice),
                    Vector{Float32}(bidVol),
                    Vector{Float32}(askPrice),
                    Vector{Float32}(askVol))
            )
    end
    res
end

function merge(datas)::SortedDict{Float64,Vector{Item}}
    res = SortedDict{Float64,Vector{Item}}()
    for data in datas
        for (time, item) in data
            if time in keys(res)
                push!(res[time], item)
            else
                res[time] = [item]
            end
        end
    end
    res
end


### Main
function backtest(market_data::SortedDict{Float64,Vector{Item}},
                  play_market::String,
                  action!::Dict{Tuple{String,String},Function})::
    SortedDict{Float64, Position}

    res = SortedDict{Float64, Position}()
    # preference
    latency = 0.050::Float64
    # MyData
    orders  = Order[]
    dealt   = Order[]
    position = Position()
    position_lst = Position()
    api_counter = Deque{Float64}();  for i in 1:300 push!(api_counter, 0.) end
    # MarketData in test
    midPrice = NaN
    current_data = Dict{Tuple{String,String}, Union{Vector{MarketActivity},MarketBook,Float64}}()
    for (time,items) in market_data
        # 各Data上をループ
        for item in items
            if (item.market, item.channel) == (play_market, "book")
                # Data更新
                current_data[item.market, item.channel] = item.data
                midPrice = (item.data.bidPrices[1] + item.data.askPrices[1]) / 2
                # 注文行動
                if (item.market, item.channel) in keys(action)
                    action![item.market, item.channel](orders, time+latency, current_data, api_counter)
                end
                # MO約定
                orders, dealt = dealing(time, orders, item.data::MarketBook)
                position = update_position(dealt, position, midPrice)
            elseif (item.market, item.channel) == (play_market, "deal")
                # Data更新
                current_data[item.market, item.channel] = item.data
                # LO約定
                orders, dealt = dealing(time, orders, item.data::Vector{MarketActivity})
                position = update_position(dealt, position, midPrice)
                current_data[item.market, item.channel] = item.data
            elseif (item.market, item.channel) in keys(action)
                # Data更新
                current_data[item.market, item.channel] = item.data
                # 注文行動
                if (item.market, item.channel) in keys(action)
                    action![item.market, item.channel](orders, time+latency, current_data, api_counter)
                end
            else
                # Data更新
                current_data[item.market, item.channel] = item.data
            end
        end # Item loop

        if (position.amount, position.pnl_open) != (position_lst.amount, position_lst.pnl_open)
            res[time] = position
        end
        position_lst = position
    end #time loop
    res
end

function count_api(api_counter::Deque{Float64}, current_time::Float64, time_theta::Float64=300)::Int32
    count_ = Int32(0)
    time = current_time - time_theta
    for elem in api_count
        count_ += (elem>time)
    end
    count_
end

function push_api!(api_counter::Deque{Float64}, time::Float64)::Deque{Float64}
    push!(api_counter, time)
    pop!(api_counter)
    api_counter
end


end  # module