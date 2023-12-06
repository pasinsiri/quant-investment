import datetime as dt
import os
import requests

# * CNN's Fear and Greed Index
# * use Firefox on MacOS as an agent
AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 12.4; rv:100.0) Gecko/20100101 Firefox/100.0"
URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

res = requests.get(URL, headers={
    "User-Agent": AGENT
})
res.raise_for_status()
res = res.json()

