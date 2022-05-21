import numpy as np 
import pandas as pd 
import requests

class FinnoFund():
    def __init__(self) -> None:
        self.base_url = 'https://api.finnomena.com/fund-service/public/api/v2/'
        self.fund_data = self.get_fund_data()[['fund_id', 'short_code']]

    def get_fund_data(self, filter_sec_active:bool = True):
        url = self.base_url + 'funds'
        res = requests.get(url) 
        if res.status_code != 200:
            raise requests.RequestException(f'request failed with status {res.status_code}')
        fund_df = pd.DataFrame(res.json()['data']) 
        if filter_sec_active:
            return fund_df[fund_df['sec_is_active'] == 1].reset_index(drop = True) 
        else:
            return fund_df 

    def get_fund_price(self, fund_name:str):
        fund_id = self.fund_data[self.fund_data['short_code'] == fund_name]['fund_id'].values[0]
        url = self.base_url + f'funds/{fund_id}/nav/q?range=MAX'
        res = requests.get(url) 
        if res.status_code != 200:
            raise requests.RequestException(f'request failed with status {res.status_code}')
        fund_df = pd.DataFrame(res.json()['data']['navs']).set_index('date')
        return fund_df 
    
    def get_multiple_fund_price(self, fund_list:list):
        all_df = None
        for fund in fund_list:
            fund_df = self.get_fund_data(fund)
            fund_df.columns = ['_'.join([fund, c]) for c in fund_df.columns]

            if all_df is None:
                all_df = fund_df 
            else:
                all_df = all_df.merge(fund_df, left_index=True, right_index=True, how='full').sort_index()
        return all_df 
