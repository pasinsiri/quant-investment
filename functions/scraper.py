import time
from selenium import webdriver

class SETScraper():
    def __init__(self, driver_type:str) -> None:
        if driver_type.lower() == 'firefox':
            self.driver = webdriver.Firefox()
        elif driver_type.lower() == 'chrome':
            self.driver = webdriver.Chrome()
        elif driver_type.lower() == 'edge':
            self.driver = webdriver.Edge()
        elif driver_type.lower() == 'safari':
            self.driver = webdriver.Safari()
        else:
            raise ValueError('driver_type is not specified')

    def get_company_information(self, ticker_list: list, sleep: int = 2):
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