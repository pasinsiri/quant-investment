import pandas as pd
import numpy as np
import requests
from ratelimit import *


class SECFundData():
    def __init__(self, api_key) -> None:
        self.headers = {
            'Ocp-Apim-Subscription-Key': api_key
        }
        self.base_url = 'https://api.sec.or.th/'

    @rate_limited(1500, 300)
    def get_amc_metadata(self):
        url = self.base_url + 'FundFactsheet/fund/amc'
        res = requests.get(url, headers=self.headers)
        return pd.DataFrame(res.json())

    def get_fund_by_amc(
            self,
            amc_uid: str,
            is_active: bool = True,
            exclude_ui: bool = True):
        url = self.base_url + f'FundFactsheet/fund/amc/{amc_uid}'
        res = requests.get(url, headers=self.headers)
        res_df = pd.DataFrame(res.json())
        if is_active:
            res_df = res_df[(res_df['fund_status'] == 'RG')].reset_index()
        if exclude_ui:
            res_df = res_df[~(res_df['proj_name_th'].str.contains(
                'ห้ามขายผู้ลงทุนรายย่อย'))].reset_index(drop=True)
        return res_df

    def get_top5_asset(self, proj_id: str, period: str):
        url = self.base_url + f'FundFactsheet/fund/{proj_id}/FundTop5/{period}'
        res = requests.get(url, headers=self.headers)
        return pd.DataFrame(res.json())
