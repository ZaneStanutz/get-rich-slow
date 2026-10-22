import re
import logging
import traceback
import time
import random
import yfinance as yf
import pandas as pd
from requests.exceptions import HTTPError, ConnectionError, Timeout, RequestException


# change to quarterly for quarterly data.
from quarterly import(
  get_revenue_growth, 
  get_earnings_growth, 
  get_fcf_divided_earnings,
  get_return_on_invested_capital,
  get_net_debt_free_cash_flow,
  get_debt_equity,
  get_reporting_basis
)

from valuation import get_valuation

from utils.db import *

from utils.df_to_pdf import *

from utils.save_df_parquet import *

from utils.notify import build_summary_email, build_failure_email, send_email

from get_tickers import update_ticker_csv


# Growth above this is usually a rebound from a weak previous period (one-off charges,
# a bad quarter) rather than a trend, so it is excluded from the screen.
MAX_EARNINGS_GROWTH = 100

# setup pandas to display all data
pd.set_option('display.max_rows', 1000)

# setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='getrich.log', encoding='utf-8', level=logging.DEBUG)


def get_stock_data(ticker: str) -> yf.Ticker:
  '''
  Get data for ticker symbol. yfinance 1.x handles auth/rate limiting internally.

  Args:
    ticker: the string ticker symbol.

  Returns:
    a yf.Ticker object.
  '''
  stock = yf.Ticker(ticker)
  logger.info(f"Created Ticker object for {ticker}")
  return stock


def process_tickers(tickers_list, batch_size=5, delay_between_batches=60):
  '''
  Process tickers in small batches with delays between batches to avoid rate limiting
  
  Args:
    tickers_list: List of ticker symbols to process
    batch_size: How many tickers to process before pausing
    delay_between_batches: How many seconds to wait between batches
  '''
  results = []
  
  for i, ticker in enumerate(tickers_list):
    try:
      # Log progress
      logger.info(f"Processing ticker {ticker} ({i+1}/{len(tickers_list)})")
      print(f"Processing ticker {ticker} ({i+1}/{len(tickers_list)})")
      
      # Skip tickers with special characters (which often cause issues)
      if re.search(r"[\^\/]", ticker):
        logger.info(f"Skipping ticker with special characters: {ticker}")
        print(f"Skipping ticker with special characters: {ticker}")
        continue
        
      stock = get_stock_data(ticker)

      # Get investment metrics
      try:
        revenue_growth = get_revenue_growth(stock)
        print(revenue_growth)
        earnings_growth = get_earnings_growth(stock)
        print(earnings_growth)
        fcf_earnings = get_fcf_divided_earnings(stock)
        print(fcf_earnings)
        roic = get_return_on_invested_capital(stock)
        print(roic)
        net_debt_fcf = get_net_debt_free_cash_flow(stock)
        print(net_debt_fcf)
        debt_equity = get_debt_equity(stock)
        print(debt_equity)
        
        # Store results
        ticker_data = {
          "ticker": ticker,
          "revenue_growth": revenue_growth,
          "earnings_growth": earnings_growth, 
          "fcf_earnings": fcf_earnings,
          "roic": roic,
          "net_debt/fcf": net_debt_fcf,
          "debt_equity" : debt_equity,
          "basis": get_reporting_basis(stock)
        }

        # Valuation is computed separately so a failure here doesn't drop the quality metrics
        try:
          valuation = get_valuation(stock)
          print(valuation)
          ticker_data.update(valuation)
        except (KeyError, IndexError, TypeError, ZeroDivisionError) as e:
          logger.error(f"Error calculating valuation for {ticker}: {str(e)}")
          print(f"Error calculating valuation for {ticker}: {str(e)}")

        results.append(ticker_data)
        print(f"Results for {ticker}: Rev Growth: {revenue_growth}%, Earnings Growth: {earnings_growth}%, FCF/Earnings: {fcf_earnings}%")
        
      except Exception as e:
        logger.error(f"Error calculating metrics for {ticker}: {str(e)}")
        print(f"Error calculating metrics for {ticker}: {str(e)}")
        
      # Add a small delay between each ticker even within a batch
      time.sleep(random.uniform(1, 3))
      
      # If we've processed a batch, take a longer pause
      if (i + 1) % batch_size == 0 and i < len(tickers_list) - 1:
        pause_time = delay_between_batches + random.uniform(-5, 5)  # Add jitter
        logger.info(f"Completed batch of {batch_size} tickers. Pausing for {pause_time:.1f} seconds...")

        #autosaving
        if results:
          print(f"Completed batch of {batch_size} tickers. saving backup.")
          results_df = pd.DataFrame(results)
          results_df.to_parquet(path=f"stock_analysis_backup.parquet", index=False)

        print(f"Completed batch of {batch_size} tickers. Auto-saving. Pausing for {pause_time:.1f} seconds...")
        time.sleep(pause_time)
        
    except Exception as e:
      logger.error(f"Unexpected error processing ticker {ticker}: {str(e)}")
      print(f"Unexpected error processing ticker {ticker}: {str(e)}")
      
  return results


def main() -> None:
  '''
  Run the full scrape: load tickers, compute metrics, filter, save results and email a summary.
  '''
  
  # Refresh the ticker list so delisted stocks drop out and new listings are included.
  # Keeps the existing csv if the download fails.
  update_ticker_csv('nasdaq_screener.csv')

  # Load ticker data
  try:
    data = pd.read_csv('nasdaq_screener.csv')
    all_tickers = data['Symbol'].to_list()
    logger.info(f"Loaded {len(all_tickers)} tickers from nasdaq_screener.csv")
    print(f"Loaded {len(all_tickers)} tickers from nasdaq_screener.csv")
  except Exception as e:
    logger.error(f"Error loading tickers: {str(e)}")
    print(f"Error loading tickers: {str(e)}")
    all_tickers = []
  
  # For testing, use a small subset of tickers
  test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
  
  # Decide which set of tickers to use (for testing or production use)
  use_test_mode = False  # Set to False to process all tickers
  tickers_to_process = test_tickers if use_test_mode else all_tickers
  
  # Process the tickers with our improved batching and rate limiting
  results = process_tickers(
    tickers_to_process, 
    batch_size=5,  # Process in small batches
    delay_between_batches=60  # Wait between batches
  )
  
  timestamp = time.strftime("%Y%m%d_%H%M%S")
  filtered_df = None
  pdf_path = None

  # Convert results to DataFrame
  if results:
    
    results_df = pd.DataFrame(results)

    # save unfiltered data.. 

    try:
      save_dataframe_to_parquet(df=results_df, folder_path='results/raw', 
                                file_name=f'raw_{timestamp}.parquet')
      print(f"Results saved to raw_{timestamp}.parquet")
    except Exception as e:
      logger.error(f"Error saving pre-filter results: {str(e)}")
      print(f"Error saving pre-filter results: {str(e)}")
  

    # Filter the results 
    # only require the quality metrics; valuation columns can be empty (e.g. negative FCF)
    quality_columns = ['revenue_growth', 'earnings_growth', 'fcf_earnings', 'roic', 'net_debt/fcf', 'debt_equity']
    df_drop = results_df.dropna(subset=quality_columns)


    filtered_df = df_drop[(df_drop['revenue_growth'] > 5) & (df_drop['earnings_growth'] > 7) & 
                      (df_drop['earnings_growth'] <= MAX_EARNINGS_GROWTH) & 
                      (df_drop['fcf_earnings'] > 80) & (df_drop['roic'] > 15) & 
                      (df_drop['net_debt/fcf'] < 5 ) & (df_drop['debt_equity'] < 80)]

    filtered_df = filtered_df.reset_index(drop=True)
    filtered_df = filtered_df.sort_values(by=['revenue_growth', 'earnings_growth', 'fcf_earnings', 'roic', 'net_debt/fcf', 'debt_equity'], 
                            ascending=[False, False, False, False, True ,True])

    # surface the cheapest quality stocks: undervalued ones on top, largest margin of safety first
    if 'margin_of_safety' in filtered_df.columns:
      filtered_df['undervalued'] = filtered_df['undervalued'].eq(True)
      filtered_df =filtered_df.sort_values(by=['undervalued', 'margin_of_safety'],
                                            ascending=[False, False], na_position='last')

    try:
      save_dataframe_to_parquet(df=filtered_df, folder_path='results/filtered',
                                file_name=f'filtered_{timestamp}.parquet')
      print(f"Results saved to stock_analysis_{timestamp}.parquet")
    except Exception as e:
      logger.error(f"Error saving filtered results: {str(e)}")
      print(f"Error saving filtered results: {str(e)}")
      
    try: 
      pdf_path = df_to_pdf(filtered_df, f"stock_analysis_{timestamp}.pdf")
      print(f"resuts saved to stock_analysis_{timestamp}.pdf")
    except Exception as e:
      logger.error(f"Error saving results to pdf: {str(e)}")
      print(f"Error saving results to pdf: {str(e)}")

  # let me know the scrape is done
  email = build_summary_email(tickers_processed=len(tickers_to_process), results_count=len(results),
                              filtered_df=filtered_df, timestamp=timestamp)
  send_email(email, attachment_path=pdf_path)


if __name__ == '__main__':
  try:
    main()
  except Exception:
    # email the crash so a long unattended run doesn't fail silently, then fail as normal
    error_traceback = traceback.format_exc()
    logger.critical(f"Scrape failed: {error_traceback}")
    send_email(build_failure_email(error_traceback, time.strftime("%Y-%m-%d %H:%M:%S")))
    raise
