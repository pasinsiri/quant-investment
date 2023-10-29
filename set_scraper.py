import json
import datetime as dt
import logging
import time
from functions.scraper import SETScraper

INDEXES_URL = {
    'set50': 'https://www.set.or.th/th/market/index/set50/overview',
    'set100': 'https://www.set.or.th/th/market/index/set100/overview',
    'sset': 'https://www.set.or.th/th/market/index/sset/overview',
    'mai': 'https://www.set.or.th/th/market/index/mai/overview',
}
EXPORT_PATH = './test/index_list/set/ticker_list.json'

tickers = {}
scraper = SETScraper(driver_type='firefox')
for index, url in INDEXES_URL.items():
    res = scraper.get_ticker_list(url=url, tag_name='div', tag_attrs={'class': 'btn-quote'})
    tickers[index] = res
    logging.info(f'Scraping {index} completed')
    time.sleep(5)

# * create an export dictionary
export = {}
export['data_date'] = dt.date.today()
export['data'] = tickers

# * save data
with open(EXPORT_PATH, 'w') as f:
    json.dump(export, f)