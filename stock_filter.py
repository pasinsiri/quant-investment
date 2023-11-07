import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

target_date = dt.date.today()
start_date = (target_date - relativedelta(years=1)).replace(day=1)
base_path = './data/prices/set/'
