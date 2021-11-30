import pandas as pd 
import numpy as np
import requests
import time
import datetime as dt
import json
import re
from bs4 import BeautifulSoup

def clean_percent(percent):
    if percent == '--':
        return 0.0
    else:
        return float(percent.replace('%', ''))

def clean_stock_name(stock, ticker):
    if ticker is not None:
        return stock.replace(ticker, '')
    else:
        return stock

def ticker_market_join(ticker, market):
    if ticker is not None and market is not None:
        return ':'.join([ticker, market])
    else:
        return None

def split_ticker(ticker, sep = ':'):
    splits = ticker.split(sep) 
    if len(splits) > 2:
        splits = [splits[0]] + [sep.join(splits[1:])]
    return splits

def pull_ft_markets(fund_source):
    res = requests.get(fund_source)
    res.encoding = 'utf-8'
    soup = BeautifulSoup(res.text, 'html.parser')

    # fund name
    fund_name = soup.find('div', {'class': 'mod-tearsheet-overview__header__container'}).text.replace('+ Add to watchlist', '').strip()

    # sectors
    sectors = soup.find('div', {'class': 'mod-weightings__sectors__table'}).find('table', {'class': 'mod-ui-table'}).findAll('tr')
    sector_df = pd.DataFrame([[fund_name] + [y.text for y in x.findAll('td')] for x in sectors], columns=['feeder_fund', 'sector', 'percent_asset', 'category_average'])
    sector_df['percent_asset'] = sector_df['percent_asset'].apply(clean_percent)
    sector_df['category_average'] = sector_df['category_average'].apply(clean_percent)
    sector_df['timestamp'] = dt.date.today()

    # regions
    regions = soup.find('div', {'class': 'mod-weightings__regions__table'}).find('table', {'class': 'mod-ui-table'}).findAll('tr')
    region_df = pd.DataFrame([[fund_name] + [y.text for y in x.findAll('td')] for x in regions], columns=['feeder_fund', 'region', 'percent_asset', 'category_average'])
    region_df['percent_asset'] = region_df['percent_asset'].apply(clean_percent)
    region_df['category_average'] = region_df['category_average'].apply(clean_percent)
    region_df['timestamp'] = dt.date.today()

    # stocks
    stock_page = soup.find('div', {'data-module-name': 'TopHoldingsApp'})
    # Stock name
    tickers = stock_page.findAll('span', {'class': 'mod-ui-table__cell__disclaimer'})
    tickers = [split_ticker(t) for t in [s.text for s in tickers]]

    # Stock ratio
    stocks = stock_page.findAll('table', {'class': 'mod-ui-table'})[1].findAll('tr')
    stock_data = [[y.text for y in x.findAll('td')][:-1] for x in stocks][:-1]


    # assert len(stock_data) == len(tickers)
    if len(stock_data) == len(tickers):
        tickers = [[fund_name] + tickers[i] + stock_data[i] for i in range(len(stock_data))]

    else:
        ticker_list = []
        # ! some tickers are missing
        for s in stock_data:
            found = 0
            for t in tickers:
                # print([s[0], t[0]])
                if t[0] in s[0]:
                    to_append = [fund_name] + t
                    to_append.extend(s)
                    ticker_list.append(to_append)
                    found = 1
                    break 
                # if ticker not found
            if found == 0:
                to_append = [fund_name] + [None, None]
                to_append.extend(s)
                ticker_list.append(to_append)
        tickers = ticker_list

    stock_df = pd.DataFrame(tickers, columns=['feeder_fund', 'ticker', 'market', 'full_name', '1yr_change', 'percent_asset'])
    stock_df['1yr_change'] = stock_df['1yr_change'].apply(clean_percent)
    stock_df['percent_asset'] = stock_df['percent_asset'].apply(clean_percent)
    stock_df['full_name'] = stock_df.apply(lambda x: clean_stock_name(x['full_name'], ticker_market_join(x['ticker'], x['market'])), axis = 1)
    stock_df['timestamp'] = dt.date.today()
    
    return sector_df, region_df, stock_df


# * Load fund quotes
with open('./keys/ft_tickers.json', 'r') as f:
    quotes = json.load(f)

# * Fetch data from FT Market
sectors = []
regions = []
stocks = []
test_count = 100
count = 0

# ? new format: add URLs and ETFs in the JSON file
for k in quotes:
    url_format = quotes[k]['url'] 
    for q in quotes[k]['lists']:
        ft_quote = quotes[k]['lists'][q]
        url = url_format.replace('[0]', ft_quote)
        tmp_sector, tmp_region, tmp_stock = pull_ft_markets(url)
        tmp_sector.insert(0, 'fund', q)
        tmp_region.insert(0, 'fund', q)
        tmp_stock.insert(0, 'fund', q)
        sectors.append(tmp_sector)
        regions.append(tmp_region)
        stocks.append(tmp_stock)
        print('Fetching {0} completed!'.format(q))

        count += 1
        if count == test_count:
            break
        time.sleep(15)


# Append all DFs
sector_df = pd.DataFrame()
region_df = pd.DataFrame()
stock_df = pd.DataFrame()

assert len(sectors) == len(regions) == len(stocks)
for i in range(len(sectors)):
    sector_df = sector_df.append(sectors[i])
    region_df = region_df.append(regions[i])
    stock_df = stock_df.append(stocks[i])

# ? Add timestamp
sector_df['timestamp'] = dt.date.today()
region_df['timestamp'] = dt.date.today()
stock_df['timestamp'] = dt.date.today()
    
# ? Save
sector_df.to_csv('./data/ft-funds/sectors.csv', index = False)
region_df.to_csv('./data/ft-funds/regions.csv', index = False)
stock_df.to_csv('./data/ft-funds/stocks.csv', index = False)
print('Scraping Completed!')
