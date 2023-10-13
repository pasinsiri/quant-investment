from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# URL of the page
url = "https://www.set.or.th/th/market/index/set50/overview"

# Initialize a web driver (make sure you have downloaded the appropriate driver)
# driver = webdriver.Chrome(executable_path="./config/chromedriver")
service = Service(executable_path="./config/chromedriver")
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=service, options=options)

