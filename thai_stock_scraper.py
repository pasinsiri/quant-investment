from functions.scraper import SiamchartScraper

export_path = ''
ss = SiamchartScraper()
info = ss.get_ticker_info_table()
