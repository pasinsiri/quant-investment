import os
from functions.scraper import SiamchartScraper

export_path = os.path.join('content/thai/ticker_list', 'thai_ticker.csv')
ss = SiamchartScraper()
info = ss.get_ticker_info_table()
info.to_csv(export_path, index=False)
print('Scraping completed')