import pandas as pd 
import numpy as np 

# ? read data scraped from Markets-FT
regions = pd.read_csv('./data/ft-funds/regions.csv')
stocks = pd.read_csv('./data/ft-funds/stocks.csv') 
sectors = pd.read_csv('./data/ft-funds/sectors.csv')

# ? my portfolio data
my_port = pd.read_csv('./meta/my_port.csv')[['fund', 'percent']]

# * my stock weights
my_stock = my_port.merge(stocks, on = 'fund', how='left')
my_stock['percent_asset'].fillna(100, inplace = True)

# TODO: enable this line if there is any mutual fund with no stock data
# my_stock['full_name'] = my_stock.apply(lambda x: x['Fund'] if pd.isna(x['full_name']) else x['full_name'], axis = 1)
my_stock['portfolio_weight'] = my_stock.apply(lambda x: x['percent'] * x['percent_asset'] / 100.0, axis = 1)

top_stocks = my_stock[['full_name', 'portfolio_weight']].groupby('full_name').sum().sort_values(by = 'portfolio_weight', ascending = False)

# * my sector weights
my_sector = my_port.merge(sectors, on = 'fund', how = 'left')
my_sector['percent_asset'].fillna(100, inplace = True)
my_sector['sector_weight'] = my_sector.apply(lambda x: x['percent'] * x['percent_asset'] / 100.0, axis = 1)

top_sector = my_sector[['sector', 'sector_weight']].groupby('sector').sum().sort_values(by = 'sector_weight', ascending=False)

# * my region weights
my_region = my_port.merge(regions, on = 'fund')
my_region['percent_asset'].fillna(100, inplace = True)
my_region['region_weight'] = my_region.apply(lambda x: x['percent'] * x['percent_asset'] / 100.0, axis = 1)
top_region = my_region[['region', 'region_weight']].groupby('region').sum().sort_values(by = 'region_weight', ascending=False)

# ? save
top_stocks.to_csv('./meta/stock_rank.csv')
top_sector.to_csv('./meta/sector_rank.csv')
top_region.to_csv('./meta/region_rank.csv')
print('Finished analysing stocks in port')