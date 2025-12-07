import os
import json
import sys
sys.path.append('..')
from functions.scraper import SiamchartScraper

export_path = os.path.join('content/thai/ticker_list', 'thai_ticker.json')
ss = SiamchartScraper()
info = ss.get_ticker_info_table(return_type='json')
# info.to_csv(export_path, index=False)
with open(export_path, 'w') as f:
    json.dump(info, f)
print('Scraping completed')