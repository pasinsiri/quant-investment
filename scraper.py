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

# Open the URL in the web driver
driver.get(url)

# Wait for the table to load (adjust the timeout as needed)
wait = WebDriverWait(driver, 10)
element = wait.until(EC.presence_of_element_located((By.ID, "table-duh0uyue18")))

# Extract the table data
table = driver.find_element(By.ID, "table-duh0uyue18")
table_html = table.get_attribute("outerHTML")

# Close the web driver
driver.quit()

# Now you can parse the table HTML using BeautifulSoup or any other method
print(table_html)
