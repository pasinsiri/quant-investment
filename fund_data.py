import numpy as np
import pandas as pd
import requests
from functions.finno_api import FinnoFund

ff = FinnoFund()
fund_data = ff.get_fund_data(filter_sec_active=True)

success_code = []

# * iterate
c = 0
nav_df = pd.DataFrame()
failed_code = []
n_funds = fund_data.shape[0]
base_url = 'https://api.finnomena.com/fund-service/public/api/v2'
for fund_id, fund_code in zip(fund_data['fund_id'], fund_data['short_code']):
    if fund_id in success_code:
        c += 1
        continue

    url = f'{base_url}/funds/{fund_id}/nav/q?range=MAX'
    res = requests.get(url)

    if res.status_code != 200:
        print(f'{fund_code} failed with status code {res.status_code}')
        failed_code.append(fund_code)
        c += 1
        continue

    if res.json()['data']['navs'] == []:
        print(f'{fund_code} (fund_id = {fund_id}) has no data')
        c += 1
        continue