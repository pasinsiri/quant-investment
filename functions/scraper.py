import time
import logging
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
        tickers = [c.text.strip('\n').strip() for c in tables]
        if driver is None:
            driver.close()
        return tickers

    def get_company_information(self, ticker_list: list, sleep: int = 2):
        driver = self._start_driver()
        res = {}

        # iterate over ticker list
        for ticker in ticker_list:
            url = f'https://www.set.or.th/th/market/product/stock/quote/{ticker.upper()}/company-profile/information'
            driver.get(url)
            data = driver.page_source
            res[ticker] = data
            logging.info(f'{ticker} is completed')
            time.sleep(sleep)
        driver.close()
        return res