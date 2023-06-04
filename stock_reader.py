import json 
import datetime as dt
import logging
import argparse
from functions.datareader import YFinanceReader

# TODO: get arguments
parser = argparse.ArgumentParser()
parser.add_argument('--s', help='Start month of (over)writing data, should be in the date format with day equal to 1')
parser.add_argument('--af', help='Annualization factor')
parser.add_argument('--ms', help='Market suffix')
parser.add_argument('--log', default='warning')

# TODO: access arguments
# START = dt.date(2022, 5, 1) # ? parsing example
# ANNUALIZATION_FACTOR = 252 # ? parsing example
# MARKET_SUFFIX = '.BK' # ? parsing example
args = parser.parse_args()
START = dt.datetime.strptime(args.s, '%Y-%m-%d')
ANNUALIZATION_FACTOR = args.af
MARKET_SUFFIX = args.ms
LOGGING_LEVEL = args.log

# * check logging config
log_level_mapping = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}
log_level = log_level_mapping[LOGGING_LEVEL.lower()]
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info(f'Retrieving data and saving since {START}.')
logging.info(f'Using annualization factor of {ANNUALIZATION_FACTOR}')
logging.info(f'The market suffix is {MARKET_SUFFIX}')

# TODO: load stock and sector data
with open('./keys/set_sectors.json', 'r') as f:
    sectors = json.load(f)

all_tickers = sectors.values()
all_tickers = [v + MARKET_SUFFIX for s in all_tickers for v in s]
logging.info(f'Getting data of {len(all_tickers)} tickers')

yfr = YFinanceReader(stock_sectors = sectors, market_suffix = MARKET_SUFFIX)
yfr.load_data(period = '1y')
yfr.save('./data/set', start_writing_date=START)