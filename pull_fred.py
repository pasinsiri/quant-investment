import pandas as pd
import pandas_datareader as pdr
import json
import datetime as dt

with open('./keys/fred_quotes.json', 'r') as f:
    fred_key = json.load(f)

start = dt.datetime(2017, 12, 29)
end = dt.datetime(2021, 10, 1)
table_dict = {}
for time in fred_key:
    time_df = dict()

    for quote in fred_key[time]:
        url_quote = fred_key[time][quote]
        try:
            tmp_df = pdr.DataReader(url_quote, 'fred', start, end)
            time_df[quote] = tmp_df[url_quote]
        except BaseException:
            print('Failed: {0}'.format(quote))

    table_dict[time] = pd.DataFrame.from_dict(time_df)

# * save to csv
for time in table_dict:
    table_dict[time].to_csv('./data/fred/' + time + '.csv', index='False')
