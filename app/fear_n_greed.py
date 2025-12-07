import datetime as dt
import os
from functions.sentiment import FearAndGreed

EXPORT_PATH = './data/sentiment/fear_n_greed'

fg = FearAndGreed()
raw = fg.fetch_data()
res_df = fg.extract_historical(raw)

curr_date = dt.date.today()
curr_date_str = dt.datetime.strftime(curr_date, '%Y-%m-%d')
res_df.to_csv(os.path.join(EXPORT_PATH, curr_date_str + '.csv'))