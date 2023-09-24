import numpy as np
import pandas as pd
import requests
import warnings
import re
from bs4 import BeautifulSoup

# * Thai Government Open Data (Web Crawling)


class ThaiGovDataReader():
    def __init__(self) -> None:
        pass

    def _get_divs(self, url: str, attr: dict = None):
        """get divs element from the given url using BeautifulSoup

        Args:
            url (str): a link
            attr (dict, optional): attributes of the div object (use to filter divs the url), can be class, id, etc.. Defaults to None.

        Returns:
            list: a list of all divs object found within the url
        """
        res = requests.get(url)
        soup = BeautifulSoup(res.content, 'html.parser')
        divs = soup.find_all('div', attr)
        return divs

    def get_tables(self, url: str, attr: dict = None) -> list:
        """get tables from the given url

        Args:
            url (str): the given url
            attr (dict, optional): the attributes of the div, e.g. class or id, that we want to extract the table from. If sets to None, the function will use the default value, which is classes "row hoverDownload" and "row hoverDownload active". Defaults to None.

        Returns:
            list: a list of table results
        """
        attr = attr or {
            'class': [
                'row hoverDownload',
                'row hoverDownload active']}
        divs = self._get_divs(url=url, attr=attr)

        # * extract url from each div
        results = list()
        for div in divs:
            # ? check if the file is in .xlsx format. if so, get the url
            div = [r for r in div]
            if div[1].find('a').find('img')['alt'] == 'xlsx':
                url = div[1].find('a')['href']
                results.append(url)

        return results

    def _get_latest_update_annual(
            self,
            url_list: list,
            regex: str,
            year_index: list) -> dict:
        """get a latest updated url for each data year

        Args:
            url_list (list): a list of all urls
            regex (str): a regex pattern to be extracted from urls. the extracted string will be used as
            year_index (list): a list of two integers indicating a start and end point of the year of the data in each url. The extracted string from this argument will be used to group the data in order to be processed further

        Returns:
            dict: a dictionary with data year (from year_index) as keys and all urls in the given year as values
        """
        assert len(
            year_index) == 2, 'year_index must contain two values, start and end indices'
        assert year_index[0] < year_index[1], 'start index must be less than end index'
        year_dict = {url: re.findall(
            regex, url)[0][year_index[0]:year_index[1]] for url in url_list}

        latest_update = dict()
        for url, year in year_dict.items():
            if year not in latest_update:
                latest_update[year] = url

        return latest_update

    def save(self, data, export_dir: str):
        """save the data in parquet format

        Args:
            data (pd.DataFrame): a pandas DataFrame
            export_dir (str): an export directory
        """
        for year, url in data.items():
            print(f'Year = {year} and URL = {url}')
            # * get file format
            file_format = url.split('.')[-1]
            response = requests.get(url)
            if response.status_code == 200:
                if file_format == 'xlsx':
                    df = pd.read_excel(response.content)
                    df.to_parquet(f'{export_dir}/{str(year)}.parquet')
                elif file_format == 'csv':
                    df = pd.read_csv(response.content)
                    df.to_parquet(f'{export_dir}/{str(year)}.parquet')
                else:
                    warnings.warn(
                        f'File format {file_format} not supported',
                        category=UserWarning)
            return
