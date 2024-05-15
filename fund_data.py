import numpy as np
import pandas as pd
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
