import json
from functions.scraper import SETScraper

INDEXES_URL = {
    'set50': 'https://www.set.or.th/th/market/index/set50/overview',
    'set100': 'https://www.set.or.th/th/market/index/set100/overview',
    'sset': 'https://www.set.or.th/th/market/index/sset/overview',
    'mai': 'https://www.set.or.th/th/market/index/mai/overview',
}

tickers = {}
scraper = SETScraper(driver_type='firefox')
for index, url in INDEXES_URL.items():
    res = scraper.get_ticker_list(url=url, tag_name='div', tag_attrs={'class': 'btn-quote'})
    tickers[index] = res