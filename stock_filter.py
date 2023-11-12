import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta

target_date = dt.date.today()
start_date = (target_date - relativedelta(years=1)).replace(day=1)
base_path = './data/prices/set/'

paths = {}
res = {}
while True:
    y, m = '{:04d}'.format(start_date.year), '{:02d}'.format(start_date.month)
    ym_path = os.path.join(base_path, y, m)
    paths[f'{y}{m}'] = ym_path

    start_date += relativedelta(months=1)
    if start_date > target_date:
        break

res = {key: pd.read_parquet(path) for key, path in paths.items()}
raw_df = pd.concat(res.values(), axis=0)

print(raw_df.head(10))