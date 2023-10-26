import time
from selenium import webdriver

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

    def get_company_information(self, ticker_list: list, sleep: int = 2):
        driver = self._start_driver()
        res = {}

        # iterate over ticker list
        for ticker in ticker_list:
            url = f'https://www.set.or.th/th/market/product/stock/quote/{ticker.upper()}/company-profile/information'
            self.driver.get(url)
            data = self.driver.page_source
            res[ticker] = data
            print(f'{ticker} is completed')
            time.sleep(sleep)
        self.driver.close()