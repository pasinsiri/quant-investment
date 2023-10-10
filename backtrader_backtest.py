"""
A backtesting code utilizing Python's backtrader library

Author: pasinsiri
Date: 2023-0722
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import pandas as pd
import backtrader as bt

# * retrieve arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--ticker',
    help='An asset ticker that will be used in backtesting'
)
parser.add_argument(
    '--start',
    help='Start date of backtesting (in YYYY-mm-dd format)'
)
args = parser.parse_args()

# * set constants
COLUMNS = ['open', 'high', 'low', 'close', 'volume']
TICKER = args.ticker
START_DATE = args.start

# * load data
price_raw = pd.read_parquet(f'./data/set/{TICKER}/price')
if len(price_raw) == 0:
    raise ValueError('Ticker does not exist')

# * filter raw data by a given date
price_filtered = price_raw[price_raw.index >= START_DATE]
if len(price_filtered) == 0:
    raise ValueError('Ticker data after a given start date does not exist')

# * select necessary columns
data = price_filtered[COLUMNS]

# TODO: create backtesting strategy


class TradingStrategy(bt.strategy.Strategy):
    params = (
        ('maperiod', 15),
    )

    def log(self, txt, date=None):
        ''' Logging function fot this strategy'''
        date = date or self.datas[0].datetime.date(0)
        print(f'{date.isoformat()}, {txt}')

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # Add a MovingAverageSimple indicator
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.maperiod)

        # Indicators for the plotting show
        bt.indicators.ExponentialMovingAverage(self.datas[0], period=25)
        bt.indicators.WeightedMovingAverage(self.datas[0], period=25,
                                            subplot=True)
        bt.indicators.StochasticSlow(self.datas[0])
        bt.indicators.MACDHisto(self.datas[0])
        rsi = bt.indicators.RSI(self.datas[0])
        bt.indicators.SmoothedMovingAverage(rsi, period=10)
        bt.indicators.ATR(self.datas[0], plot=False)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}'
                )
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log(
                    f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm {order.executed.comm:.2f}'
                )

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log(
            f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log(f'Close, {self.dataclose[0]:.2f}')

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # Not yet ... we MIGHT BUY if ...
            if self.dataclose[0] > self.sma[0]:

                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log(f'BUY CREATE, {self.dataclose[0]:.2f}')

                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()

        else:

            if self.dataclose[0] < self.sma[0]:
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log(f'SELL CREATE, {self.dataclose[0]:.2f}')

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()


if __name__ == '__main__':

    # ingest data to backtrader environment
    data = bt.feeds.PandasData(dataname=price_filtered)

    # initialize Cerebro engine
    cerebro = bt.Cerebro()

    # add a strategy
    cerebro.addstrategy(TradingStrategy)

    # add the Data Feed to Cerebro
    cerebro.adddata(data)

    # set investing amount
    cerebro.broker.setcash(1e6)

    # add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)

    # set the commission - 0.1% ... divide by 100 to remove the %
    cerebro.broker.setcommission(commission=0.001)

    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')

    cerebro.run()

    print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')

    cerebro.plot()
