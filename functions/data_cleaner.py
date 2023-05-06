import pandas as pd
import numpy as np 

class OHLCVCleaner():
    def __init__(self, ohlcv) -> None:
        self.ohlcv = ohlcv

    def _gen_new_col_name(self, col:str, prefix:str, replace:bool, delimiter:str = '_'):
        return col if replace else delimiter.join([prefix, col])

    def _adjust_ohlc_single_ticker(self, ticker_df, replace:bool):
        # sort df by date, descending
        ticker_df.sort_index(ascending=False, inplace=True)

        # fillna: 0 for dividends and 1 for stock splits
        ticker_df.fillna({'dividends': 0, 'stock splits': 1}, inplace = True)
        # for stock split, replace 0 with 1 since we'll use cumulative product
        ticker_df['stock splits'].apply(lambda x: 1 if x == 0 else x)

        # calculate accumulated dividends and stock splits
        ticker_df['accum_dividends'] = np.cumsum(ticker_df['dividends'])
        ticker_df['accum_stock splits'] = np.cumprod(ticker_df['stock splits'])

        for col in ['open', 'high', 'low', 'close']:
            new_col = self._gen_new_col_name(col=col, prefix='adj_', replace=replace)
            ticker_df[new_col] = ticker_df.apply(lambda row: (row['close'] * row['accum_stock splits']) + row['accum_dividends'], axis = 1)

        return ticker_df
    
    def adjust_ohlc(self, ohlc, replace:bool = False):
        # assert columns
        cols = ['open', 'high', 'low', 'close', 'stock splits', 'dividends']
        unexisted = [c for c in cols if c not in ohlc.columns]
        assert len(unexisted) == 0, f'{", ".join(unexisted)} not found in the OHLC dataframe'

        tickers = ohlc.index.get_level_values(1).unique()

        new_df = pd.DataFrame()
        for ticker in tickers:
            ticker_df = ohlc[ohlc.index.get_level_values(1) == ticker]
            adjusted_ticker_df = self._adjust_ohlc_single_ticker(ticker_df=ticker_df, replace=replace)
            new_df = pd.concat([new_df, adjusted_ticker_df])
        return new_df