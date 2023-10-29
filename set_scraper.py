import requests
from bs4 import BeautifulSoup
from functions.scraper import SETScraper

URL_LIST = [
    'https://www.set.or.th/th/market/index/set50/overview',
    'https://www.set.or.th/th/market/index/set100/overview',
    'https://www.set.or.th/th/market/index/sset/overview',
    'https://www.set.or.th/th/market/index/mai/overview'
]