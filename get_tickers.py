''' This is a module for programattically updating the csv
  stock ticker data everytime the script is run

  Downloads every stock listed on NASDAQ, NYSE and AMEX from the NASDAQ
  stock screener and saves it in the same format as a manual download
  from https://www.nasdaq.com/market-activity/stocks/screener
'''
import logging
import os

import pandas as pd
import requests


logger = logging.getLogger(__name__)

SCREENER_URL = 'https://api.nasdaq.com/api/screener/stocks'
TICKER_CSV = 'nasdaq_screener.csv'
# The API rejects requests without a browser like user agent
REQUEST_HEADERS = {
  'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/124.0 Safari/537.36',
  'Accept': 'application/json',
}
# A download with fewer rows than this is treated as broken rather than replacing the list
MIN_EXPECTED_TICKERS = 3000


def download_tickers() -> pd.DataFrame:
  '''
  Download the current list of US listed stocks from the NASDAQ screener.

  Returns:
    A DataFrame with the same columns as the manual CSV download
    (Symbol, Name, Last Sale, Net Change, % Change, Market Cap, Country, IPO Year, Volume, Sector, Industry).
  '''
  response = requests.get(SCREENER_URL, params={'tableonly': 'true', 'download': 'true'},
                          headers=REQUEST_HEADERS, timeout=(15, 60))
  response.raise_for_status()

  data = response.json()['data']
  headers = data['headers']
  tickers = pd.DataFrame(data['rows'])

  # rename api keys to the csv column names and drop the website link column
  tickers = tickers[[key for key in headers if key != 'url']].rename(columns=headers)
  return tickers


def update_ticker_csv(csv_path: str = TICKER_CSV) -> bool:
  '''
  Refresh the ticker csv with the latest listings.

  The existing file is only replaced after a complete download, so a network
  problem leaves the previous list in place.

  Args:
    csv_path: where to save the ticker list

  Returns:
    True if the csv was updated, False if the existing file was kept.
  '''
  try:
    tickers = download_tickers()
  except (requests.RequestException, KeyError, ValueError) as e:
    logger.error(f"Error downloading ticker list, keeping existing {csv_path}: {str(e)}")
    print(f"Error downloading ticker list, keeping existing {csv_path}: {str(e)}")
    return False

  if len(tickers) < MIN_EXPECTED_TICKERS:
    logger.error(f"Ticker download only returned {len(tickers)} rows, keeping existing {csv_path}")
    print(f"Ticker download only returned {len(tickers)} rows, keeping existing {csv_path}")
    return False

  # write to a temp file first so a crash mid write can't corrupt the list
  temp_path = f"{csv_path}.tmp"
  tickers.to_csv(temp_path, index=False)
  os.replace(temp_path, csv_path)

  logger.info(f"Updated {csv_path} with {len(tickers)} tickers")
  print(f"Updated {csv_path} with {len(tickers)} tickers")
  return True


if __name__ == '__main__':
  update_ticker_csv()
