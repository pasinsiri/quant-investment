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
