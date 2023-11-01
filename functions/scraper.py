import pandas as pd
import time
import logging
import re
import requests
from selenium import webdriver
from bs4 import BeautifulSoup

class SETScraper():
    def __init__(self, driver_type:str) -> None:
        self.driver_type = driver_type

    def _start_driver(self):
        if self.driver_type.lower() == 'firefox':
            return webdriver.Firefox()
        elif self.driver_type.lower() == 'chrome':
            return webdriver.Chrome()
        elif self.driver_type.lower() == 'edge':
            return webdriver.Edge()
        elif self.driver_type.lower() == 'safari':
            return webdriver.Safari()
        else:
            raise ValueError('driver_type is not specified')
        
    def get_ticker_list(self, url: str, tag_name: str, tag_attrs: dict, driver = None):
        if driver is None:
            driver = self._start_driver()
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, features='lxml')
        tables = soup.find_all(tag_name, attrs=tag_attrs)
        tickers = [c.text.strip('\n').strip().split('\n')[0] for c in tables]
        if driver is None:
            driver.close()
        return tickers

    def get_company_information(self, ticker_list: list, sleep: int = 1):
        driver = self._start_driver()
        res = {}

        # iterate over ticker list
        for ticker in ticker_list:
            url = f'https://www.set.or.th/th/market/product/stock/quote/{ticker.upper()}/company-profile/information'
            driver.get(url)
            raw = driver.page_source

            # ? if ticker's information is not found, i.e. the current url is different from the parsed url, return None
            if url != driver.current_url:
                res[ticker] = None
                continue

            soup = BeautifulSoup(raw, features='lxml')
            ticker_info = soup.find_all('span', attrs={'class': 'mb-3'})[0].text.strip('\n').strip()
            res[ticker] = ticker_info
            logging.info(f'{ticker} is completed')
            time.sleep(sleep)
        driver.close()
        return res
    
class SiamchartScraper():
    def __init__(self) -> None:
        self.url = 'http://siamchart.com/stock/'

    def get_ticker_info_table(self):
        TICKER_PATTERN = r'[\w\d]*\('
        INFO_PATTERN = r'\[.*\]'

        # * get request
        res = requests.get(self.url)
        soup = BeautifulSoup(res.text)
        links = soup.find_all('a', href=lambda href: href and 'stock-chart' in href)
        # * extract ticker and info
        ticker_list = [re.findall(TICKER_PATTERN, link.text)[0][:-1] for link in links]
        info_list_raw = [re.findall(INFO_PATTERN, link.text) for link in links]

        info_list = []
        for ticker, info in zip(ticker_list, info_list_raw):
            if len(info) == 0:
                info_list_raw.append([ticker])
                continue
            info = info[0].strip(r'\[\]').split(',')
            info_list.append([ticker, info])
        info_df = pd.DataFrame(info_list, columns=['ticker', 'tag'])
        return info_df