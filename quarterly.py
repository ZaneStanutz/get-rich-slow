'''This is a module providing functions to analyze the
   Value buy potential of a certain stock based off a quarterly reporting period

   Results are annualized so they can be compared against the same
   thresholds as yearly.py:
   - Flow items (income, cash flow) use trailing twelve months (TTM),
     the sum of the last 4 quarters.
   - Balance sheet items use the latest quarter.
   - Growth compares the latest quarter to the same quarter one year earlier,
     which avoids seasonal swings between neighbouring quarters.

   Companies without enough quarterly history (e.g. foreign companies that
   report every six months) fall back to the yearly.py version of every metric.
'''
import functools
from typing import Callable, Optional

import yfinance as yf
import pandas as pd

import yearly


# Index of the same quarter one year earlier (columns are newest first)
SAME_QUARTER_LAST_YEAR = 4


def percent_round(percentage:float) -> float:
  """Helper function to round percent values

  Rounds all inputs to 2 decimal places.

  Args:
      percentage: the floating point percentage to round.

  Returns:
      float: the rounded floating point number.
  """
  return round(percentage, 2);


def drop_incomplete_periods(statement: pd.DataFrame) -> pd.DataFrame:
  """Remove periods (columns) that are mostly empty.

  Yahoo sometimes lists the newest period before its data is loaded,
  e.g. only 4 of 46 rows filled. Treating that as the latest period breaks every metric.

  Args:
      statement: a yfinance financial statement (rows = line items, columns = periods)

  Returns:
      The statement with columns under half as complete as the fullest column removed.
  """
  if statement.empty:
    return statement

  filled = statement.notna().sum()
  return statement.loc[:, filled >= filled.max() * 0.5]


def get_latest_value(statement: pd.DataFrame, row: str, default: float = None, period: int = 0) -> float:
  """Get a period's value for a row of a financial statement.

  Companies report different line items, so a row can be missing entirely
  or be NaN for a period. Mostly empty periods are skipped.

  Args:
      statement: a yfinance financial statement (rows = line items, columns = periods)
      row: the line item to look up, e.g. 'Total Debt'
      default: value to use when the row is missing or NaN. If None, a KeyError is raised.
      period: which period to read, 0 = latest, 1 = the one before, etc.

  Returns:
      The value as a float.
  """
  statement = drop_incomplete_periods(statement)

  if (not statement.empty and row in statement.index and len(statement.columns) > period
      and pd.notna(statement.loc[row].iloc[period])):
    return float(statement.loc[row].iloc[period])

  if default is None:
    raise KeyError(f"'{row}' missing for period {period} — statement may be empty (possible rate limit)")

  return default


def get_earnings(statement: pd.DataFrame, periods: list) -> list:
  """Get earnings for several periods, excluding one-time items where possible.

  Uses 'Normalized Income' (net income without unusual items such as
  gains on investments), falling back to 'Net Income' when it isn't reported.
  All periods come from the same row so growth never mixes the two definitions.

  Args:
      statement: a yfinance income statement
      periods: the periods to read, 0 = latest, 1 = the one before, etc.

  Returns:
      A list of earnings values in the same order as periods.
  """
  try:
    return [get_latest_value(statement, 'Normalized Income', period=period) for period in periods]
  except KeyError:
    return [get_latest_value(statement, 'Net Income', period=period) for period in periods]


def get_ttm_value(statement: pd.DataFrame, row: str) -> float:
  """Get the trailing twelve months (TTM) value of a flow item.

  Sums the 4 most recent quarters, turning a quarterly figure into a yearly one.

  Args:
      statement: a yfinance quarterly income or cash flow statement
      row: the line item to sum, e.g. 'Net Income'

  Returns:
      The TTM value as a float.
  """
  quarters = [get_latest_value(statement, row, period=quarter) for quarter in range(4)]
  return sum(quarters)


def get_year_over_year_growth(statement: pd.DataFrame, row: str) -> Optional[float]:
  """Growth of the latest quarter versus the same quarter one year earlier.

  Args:
      statement: a yfinance quarterly financial statement
      row: the line item to compare, e.g. 'Total Revenue'

  Returns:
      A floating point percentage, or None if last year's value is not positive
      (growth from a loss is not meaningful).
  """
  current_period = get_latest_value(statement, row)
  previous_period = get_latest_value(statement, row, period=SAME_QUARTER_LAST_YEAR)

  return get_growth(current_period, previous_period)


def get_growth(current_period: float, previous_period: float) -> Optional[float]:
  """Percentage growth from a previous period to the current one.

  Args:
      current_period: the latest value
      previous_period: the value being compared against

  Returns:
      A floating point percentage, or None if the previous value is not positive
      (growth from a loss is not meaningful).
  """
  if previous_period <= 0:
    return None

  growth = (current_period - previous_period) / previous_period
  return percent_round(growth*100)



def has_quarterly_history(stock: yf.Ticker) -> bool:
  """Check if a stock has enough quarterly data for the quarterly metrics.

  Year over year growth needs 5 quarters of income statement and TTM
  free cash flow needs 4 quarters of cash flow statement.

  Args:
      stock: the stock in question as a yf.Ticker object

  Returns:
      True if the quarterly metrics can be calculated.
  """
  income_quarters = len(drop_incomplete_periods(stock.quarterly_income_stmt).columns)
  cash_flow_quarters = len(drop_incomplete_periods(stock.quarterly_cash_flow).columns)

  return income_quarters > SAME_QUARTER_LAST_YEAR and cash_flow_quarters >= 4


def get_reporting_basis(stock: yf.Ticker) -> str:
  """Which data the metrics for a stock are based on.

  Args:
      stock: the stock in question as a yf.Ticker object

  Returns:
      'quarterly', or 'yearly' if the stock falls back to yearly data.
  """
  return 'quarterly' if has_quarterly_history(stock) else 'yearly'


def yearly_fallback(metric: Callable) -> Callable:
  """Decorator that uses the yearly.py metric of the same name when quarterly data is missing.

  The choice is made per stock, so every metric for one stock uses the same basis.

  Args:
      metric: a quarterly metric function taking a yf.Ticker

  Returns:
      The wrapped metric function.
  """
  yearly_metric = getattr(yearly, metric.__name__)

  @functools.wraps(metric)
  def wrapper(stock: yf.Ticker):
    if has_quarterly_history(stock):
      return metric(stock)
    return yearly_metric(stock)

  return wrapper


@yearly_fallback
def get_return_on_equity(stock:yf.Ticker):
  """Get the ROE (return on equity) of a stock using TTM net income.

  Args:
      stock: the stock in question as a yf.Ticker object

  Returns:
      Float: The return on equity as a floating point number

  """
  balance_sheet = stock.quarterly_balance_sheet
  total_assets = get_latest_value(balance_sheet, 'Total Assets')
  total_liabilities = get_latest_value(balance_sheet, 'Total Liabilities Net Minority Interest')

  net_income = sum(get_earnings(stock.quarterly_income_stmt, range(4)))

  # calculate shareholder equity (total assets - total liability)
  shareholder_equity = total_assets - total_liabilities

  # calculate ROE ( net income / sharholders equity )
  return_on_equity = net_income / shareholder_equity
  return percent_round(return_on_equity)

@yearly_fallback
def get_revenue_growth(stock:yf.Ticker) -> Optional[float]:
  """calculates year over year revenue growth of a stock

  Compares the latest quarter to the same quarter one year earlier.
  A buy is > 5%

  Args:
      stock: the stock in question as a yf.Ticker object
  Retuns:
      A floating point percentage representing revenue growth.
  """
  return get_year_over_year_growth(stock.quarterly_income_stmt, 'Total Revenue')

@yearly_fallback
def get_earnings_growth(stock:yf.Ticker) -> Optional[float]:
  """
  Calculates year over year earnings growth of a stock

  Compares the latest quarter's normalized income to the same quarter one year earlier.
  A buy is > 7 %.

  Args:
      stock: the stock in question as a yf.Ticker object
  Retuns:
      A floating point percentage representing earnigs growth.
  """
  current_period, previous_period = get_earnings(stock.quarterly_income_stmt, [0, SAME_QUARTER_LAST_YEAR])
  return get_growth(current_period, previous_period)


@yearly_fallback
def get_fcf_divided_earnings(stock: yf.Ticker) -> Optional[float]:
  """
  Calculates TTM free cash flow divided by TTM normalized earnings.

  A buy is > 80%.

  Args:
    stock: the stock in question as a yf.Ticker object

  Returns:
    A floating point percentage, or None if earnings are not positive
    (a negative FCF over negative earnings would look like a healthy ratio).
  """

  # Free Cash Flow (FCF) = Net Cash from Operations - Capital Expenditures
  free_cash_flow = get_ttm_value(stock.quarterly_cash_flow, 'Free Cash Flow')
  net_earnings = sum(get_earnings(stock.quarterly_income_stmt, range(4)))

  if net_earnings <= 0:
    return None

  return percent_round(free_cash_flow/net_earnings*100)


@yearly_fallback
def get_return_on_invested_capital(stock:yf.Ticker) -> float :
  '''
  Calculates the return on invested capital (ROIC) over the trailing twelve months
  a buy is > 15%

  NOPAT = TTM Operating Income x (1 - Tax Rate)
  Invested Capital = Total Equity + Total Debt - Cash and Cash Equivalents
  ROIC = NOPAT ÷ Invested Capital

  Total Debt from yfinance already includes capital lease obligations,
  so leases are not added separately.

  Args:
    stock: the stock in question as a yf.Ticker object

  Returns:
    A floating point percentage.
  '''
  income_statement = stock.quarterly_income_stmt

  if income_statement.empty:
    raise KeyError(f"income statement is empty (possible rate limit)")

  # Fall back to the 21% US corporate rate when no tax rate is reported
  tax_rate = get_latest_value(income_statement, 'Tax Rate For Calcs', default=0.21)
  operating_income = get_ttm_value(income_statement, 'Operating Income')
  no_pat = operating_income * (1 - tax_rate)

  balance_sheet = stock.quarterly_balance_sheet
  total_equity = get_latest_value(balance_sheet, 'Total Equity Gross Minority Interest')
  # A missing debt or cash row means the company reports none
  debt = get_latest_value(balance_sheet, 'Total Debt', default=0.0)
  cash = get_latest_value(balance_sheet, 'Cash And Cash Equivalents', default=0.0)

  invested_capital = total_equity + debt - cash

  if invested_capital <= 0:
    raise ZeroDivisionError(f"invested capital is not positive ({invested_capital}), ROIC is not meaningful")

  return percent_round(no_pat/invested_capital*100)



@yearly_fallback
def get_net_debt_free_cash_flow(stock:yf.Ticker) -> Optional[float]:
  '''
  Calculates how many years of TTM free cash flow it would take to pay off net debt.

  Net Debt = Total Debt - Cash and Cash Equivalents
  A buy is < 5

  Args:
    stock: the stock in question as a yf.Ticker object

  Returns:
    A floating point ratio, or None if FCF is not positive
    (a cash burning company would otherwise get a negative ratio and pass the screen).
  '''
  free_cash_flow = get_ttm_value(stock.quarterly_cash_flow, 'Free Cash Flow')

  if free_cash_flow <= 0:
    return None

  balance_sheet = stock.quarterly_balance_sheet
  # A missing debt or cash row means the company reports none
  total_debt = get_latest_value(balance_sheet, 'Total Debt', default=0.0)
  cash = get_latest_value(balance_sheet, 'Cash And Cash Equivalents', default=0.0)

  net_debt = total_debt - cash

  return percent_round(net_debt/free_cash_flow)


@yearly_fallback
def get_debt_equity(stock:yf.Ticker) -> Optional[float]:
  '''
  Calculates total debt as a percentage of shareholder equity.

  Equity = Total Assets - Total Liabilities
  A buy is < 80%

  Args:
    stock: the stock in question as a yf.Ticker object

  Returns:
    A floating point percentage, or None if equity is not positive
    (liabilities exceed assets, which would give a negative ratio that passes the screen).
  '''
  balance_sheet = stock.quarterly_balance_sheet
  # A missing debt row means the company reports none
  total_debt = get_latest_value(balance_sheet, 'Total Debt', default=0.0)
  total_liabilites = get_latest_value(balance_sheet, 'Total Liabilities Net Minority Interest')
  total_assets = get_latest_value(balance_sheet, 'Total Assets')

  equity = total_assets - total_liabilites

  if equity <= 0:
    return None

  return percent_round(total_debt/equity*100)
