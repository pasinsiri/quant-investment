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
        
    def _extract_ticker_siamchart(self, pattern: str, text: str):
        matches = [re.findall(pattern, t.text)[-1] for t in text]
        
    def scrape_siamchart(self):
        URL = 'http://siamchart.com/stock/'
        res = requests.get(URL)
        soup = BeautifulSoup(res.text)
        
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