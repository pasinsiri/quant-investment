import numpy as np 
import pandas as pd 
import requests
import warnings 
import re 
from bs4 import BeautifulSoup

# * Thai Government Open Data (Web Crawling)
class ThaiGovDataReader():
    def __init__(self, base_url:str) -> None:
        self.base_url = base_url