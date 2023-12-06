import datetime as dt
import os
import pandas as pd
import requests

# * CNN's Fear and Greed Index
class FearAndGreed():
    def __init__(self) -> None:
        # ? use Firefox on MacOS as an agent
        self.agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 12.4; rv:100.0) Gecko/20100101 Firefox/100.0"
        self.url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

    def fetch_data(self):
        res = requests.get(self.url, headers={
            "User-Agent": self.agent
        })
        res.raise_for_status()
        res = res.json()
        return res
    
    def extract_historical(self, raw_data: dict):
        historical = raw_data['fear_and_greed_historical']['data']
        historical_df = pd.DataFrame(historical)
        historical_df['date'] = historical_df['x'].apply(lambda x: dt.datetime.fromtimestamp(x / 1000.0).date())
        historical_df = historical_df.rename(columns={'y': 'fear_n_greed_index'})
        
        res_df = historical_df[['date', 'fear_n_greed_index', 'rating']].set_index('date')
        return res_df
    
if __name__ == '__main__':
    fg = FearAndGreed()
    raw = fg.fetch_data()
    res_df = fg.extract_historical(raw)
    # print(res_df.head())
