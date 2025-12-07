import json
import datetime as dt
import logging
import time
import sys
sys.path.append('..')
from functions.scraper import SETScraper

# INDEXES_URL = {
#     'set50': 'https://www.set.or.th/th/market/index/set50/overview',
#     'set100': 'https://www.set.or.th/th/market/index/set100/overview',
#     'sset': 'https://www.set.or.th/th/market/index/sset/overview',
#     'mai_agro': 'https://www.set.or.th/th/market/index/mai/agro',
#     'mai_consump': 'https://www.set.or.th/th/market/index/mai/consump',
#     'mai_fincial': 'https://www.set.or.th/th/market/index/mai/fincial',
#     'mai_indus': 'https://www.set.or.th/th/market/index/mai/indus',
#     'mai_propcon': 'https://www.set.or.th/th/market/index/mai/propcon',
#     'mai_resourc': 'https://www.set.or.th/th/market/index/mai/resourc',
#     'mai_service': 'https://www.set.or.th/th/market/index/mai/service',
#     'mai_tech': 'https://www.set.or.th/th/market/index/mai/tech'
# }

INDEXES_URL = {
    'set50': 'https://www.set.or.th/th/market/index/set50/overview',
    'set100': 'https://www.set.or.th/th/market/index/set100/overview',
    'sset': 'https://www.set.or.th/th/market/index/sset/overview'
}
EXPORT_PATH = './test/index_list/set/ticker_list.json'

tickers = {}
scraper = SETScraper(driver_type='firefox')
driver = scraper._start_driver()
for index, url in INDEXES_URL.items():
    res = scraper.get_ticker_list(url=url, tag_name='div', tag_attrs={'class': 'btn-quote'}, driver=driver)
    tickers[index] = res
    logging.info(f'Scraping {index} completed')
    time.sleep(5)
driver.close()

# * create an export dictionary
export = {}
export['data_date'] = dt.datetime.strftime(dt.date.today(), '%Y-%m-%d')
export['data'] = tickers

# * save data
with open(EXPORT_PATH, 'w') as f:
    json.dump(export, f)