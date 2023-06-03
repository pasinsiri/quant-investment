import json 
import datetime as dt 
from functions.datareader import YFinanceReader

# TODO: set parameters
start = dt.date(2022, 4, 1)
ANNUALIZATION_FACTOR = 252

# TODO: load stock and sector data
with open('./keys/set_sectors.json', 'r') as f:
    sectors = json.load(f)

all_tickers = sectors.values()
all_tickers = [v + '.BK' for s in all_tickers for v in s]

yfr = YFinanceReader(stock_sectors = sectors, market_suffix = '.BK')
yfr.load_data(period = '1m')
yfr.save('./data/set', start_writing_date=dt.datetime(2022, 4, 1))