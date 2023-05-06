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

        # fillna: 0 for dividend and 1 for split
        ticker_df.fillna({'dividend': 0, 'split': 1}, inplace = True)

        # calculate accumulated dividend and split
        ticker_df['accum_dividend'] = np.cumsum(ticker_df['dividend'])
        ticker_df['accum_split'] = np.cumprod(ticker_df['split'])

        for col in ['open', 'high', 'low', 'close']:
            new_col = self._gen_new_col_name(col=col, prefix='adj_', replace=replace)
            ticker_df[new_col] = ticker_df.apply(lambda row: (row['close'] * row['accum_split']) + row['accum_dividend'], axis = 1)

        return ticker_df
    
    def adjust_ohlc(self, ohlc, replace:bool = False):
        # assert columns
        cols = ['open', 'high', 'low', 'close', 'split', 'dividend']
        unexisted = [c for c in cols if c not in ohlc.columns]
        assert len(unexisted) == 0, f'{", ".join(unexisted)} not found in the OHLC dataframe'

        tickers = list(set(ohlc.index.get_level_values(1)))

        new_df = pd.DataFrame()
        for ticker in tickers:
            ticker_df = ohlc.loc[:, ticker]
            adjusted_ticker_df = self._adjust_ohlc_single_ticker(ticker_df=ticker_df, replace=replace)
            new_df = pd.concat([new_df, adjusted_ticker_df])
        return new_df