'''This is a module providing functions to estimate the intrinsic value
   of a stock and compare it to the current market price.

   The main model is a two-stage discounted cash flow (DCF) on free cash flow.
   A Graham number and FCF yield are included as simple cross-checks.
'''
import math
from typing import Optional

import yfinance as yf
import pandas as pd

import yearly


# Required annual return used to discount future cash flows.
DISCOUNT_RATE = 0.10
# Long-run growth after the projection period, roughly inflation / GDP.
TERMINAL_GROWTH = 0.025
# Years of explicit cash flow projection. Growth fades to TERMINAL_GROWTH
# over the second half of the period.
PROJECTION_YEARS = 10
# Cap on the starting growth rate so one great year doesn't explode the value.
MAX_GROWTH = 0.15
# Number of most recent years of FCF averaged for the starting cash flow (see get_base_fcf).
FCF_AVERAGE_YEARS = 3
# Range of growth rates searched when solving for the growth the price implies.
IMPLIED_GROWTH_RANGE = (-0.5, 2.0)
# Minimum discount to intrinsic value before calling a stock undervalued.
MARGIN_OF_SAFETY_TARGET = 30
# Sectors where free cash flow doesn't measure owner earnings (client money
# flowing through banks, brokers, insurers and asset managers), so a FCF based DCF is skipped.
UNRELIABLE_FCF_SECTORS = ['Financial Services']

# Extra discount rate added for companies outside the US, for currency, political
# and legal risk (e.g. China's VIE share structures). These are rough tiers,
# loosely modelled on published country risk premiums; adjust them to taste.
DEVELOPED_PREMIUM = 0.01
ADVANCED_EMERGING_PREMIUM = 0.02
EMERGING_PREMIUM = 0.035
HIGH_RISK_PREMIUM = 0.05
# Used for any non US country or currency not listed below
DEFAULT_FOREIGN_PREMIUM = EMERGING_PREMIUM

COUNTRY_RISK_PREMIUMS = {
  'United States': 0.0,
  **dict.fromkeys(['Canada', 'United Kingdom', 'Ireland', 'Germany', 'France', 'Netherlands',
                   'Belgium', 'Luxembourg', 'Switzerland', 'Austria', 'Sweden', 'Norway', 'Denmark',
                   'Finland', 'Australia', 'New Zealand', 'Japan', 'Singapore', 'Bermuda',
                   'Cayman Islands', 'Jersey', 'Guernsey', 'Monaco', 'Spain', 'Italy', 'Portugal'],
                  DEVELOPED_PREMIUM),
  **dict.fromkeys(['Israel', 'South Korea', 'Taiwan', 'Hong Kong', 'Chile', 'Poland', 'Cyprus',
                   'Greece', 'Malta'],
                  ADVANCED_EMERGING_PREMIUM),
  **dict.fromkeys(['India', 'Indonesia', 'Mexico', 'Malaysia', 'Thailand', 'Philippines', 'Peru',
                   'Colombia', 'Uruguay', 'Panama', 'Kazakhstan', 'United Arab Emirates', 'Vietnam'],
                  EMERGING_PREMIUM),
  **dict.fromkeys(['China', 'Brazil', 'South Africa', 'Turkey', 'Argentina', 'Nigeria', 'Egypt',
                   'Russia', 'Ukraine', 'Pakistan'],
                  HIGH_RISK_PREMIUM),
}

CURRENCY_RISK_PREMIUMS = {
  'USD': 0.0,
  **dict.fromkeys(['CAD', 'GBP', 'EUR', 'CHF', 'SEK', 'NOK', 'DKK', 'AUD', 'NZD', 'JPY', 'SGD'],
                  DEVELOPED_PREMIUM),
  **dict.fromkeys(['ILS', 'KRW', 'TWD', 'HKD', 'CLP', 'PLN'], ADVANCED_EMERGING_PREMIUM),
  **dict.fromkeys(['INR', 'IDR', 'MXN', 'MYR', 'THB', 'PHP', 'PEN', 'COP', 'AED', 'VND'],
                  EMERGING_PREMIUM),
  **dict.fromkeys(['CNY', 'BRL', 'ZAR', 'TRY', 'ARS', 'NGN', 'EGP', 'RUB'], HIGH_RISK_PREMIUM),
}

# Exchange rates fetched this run, keyed by (from, to) currency
_fx_cache = {}


def percent_round(percentage: float) -> float:
  """Helper function to round percent values

  Args:
      percentage: the floating point percentage to round.

  Returns:
      float: the rounded floating point number.
  """
  return round(percentage, 2)


def get_free_cash_flow_history(stock: yf.Ticker) -> pd.Series:
  """Get the yearly free cash flow history, most recent year first.

  Args:
      stock: the stock in question as a yf.Ticker object

  Returns:
      A pandas Series of free cash flow values with NaN years removed.
  """
  cash_flow = stock.cashflow

  if cash_flow.empty or 'Free Cash Flow' not in cash_flow.index:
    raise KeyError(f"'Free Cash Flow' not found — cash flow statement may be empty (possible rate limit)")

  return cash_flow.loc['Free Cash Flow'].dropna()


def get_current_price(stock: yf.Ticker) -> float:
  """Get the current share price.

  Args:
      stock: the stock in question as a yf.Ticker object

  Returns:
      The current share price as a float.
  """
  info = stock.info
  price = info.get('currentPrice') or info.get('regularMarketPrice')

  if not price:
    raise KeyError(f"'currentPrice' not found in stock info")

  return float(price)


def get_shares_outstanding(stock: yf.Ticker) -> float:
  """Get the number of shares outstanding.

  Args:
      stock: the stock in question as a yf.Ticker object

  Returns:
      The number of shares outstanding as a float.
  """
  shares = stock.info.get('sharesOutstanding')

  if not shares:
    balance_sheet = stock.balance_sheet
    if 'Ordinary Shares Number' in balance_sheet.index:
      shares = balance_sheet.loc['Ordinary Shares Number'].dropna().iloc[0]

  if not shares:
    raise KeyError(f"'sharesOutstanding' not found")

  return float(shares)


def get_market_cap(stock: yf.Ticker) -> float:
  """Get the market capitalization in the trading currency.

  Prefers Yahoo's market cap, which covers every share class and is already
  per ADR for foreign listings. Falls back to price x shares outstanding.

  Args:
      stock: the stock in question as a yf.Ticker object

  Returns:
      The market cap as a float.
  """
  market_cap = stock.info.get('marketCap')

  if not market_cap:
    market_cap = get_current_price(stock) * get_shares_outstanding(stock)

  return float(market_cap)


def get_fx_rate(from_currency: str, to_currency: str) -> float:
  """Get the exchange rate to convert one currency into another.

  Rates are cached so each currency pair is only fetched once per run.

  Args:
      from_currency: the currency to convert from, e.g. 'CNY'
      to_currency: the currency to convert to, e.g. 'USD'

  Returns:
      How many units of to_currency one unit of from_currency is worth.
  """
  if from_currency == to_currency:
    return 1.0

  pair = (from_currency, to_currency)
  if pair not in _fx_cache:
    rate = yf.Ticker(f"{from_currency}{to_currency}=X").fast_info['last_price']
    if not rate or pd.isna(rate):
      raise KeyError(f"exchange rate {from_currency}->{to_currency} not available")
    _fx_cache[pair] = float(rate)

  return _fx_cache[pair]


def get_currency_factor(stock: yf.Ticker) -> float:
  """Get the factor converting financial statement values into the trading currency.

  Foreign companies listed in the US often report statements in their home
  currency (e.g. CNY, IDR) while the share price is in USD.

  Args:
      stock: the stock in question as a yf.Ticker object

  Returns:
      The multiplier for statement values, 1.0 when the currencies match.
  """
  info = stock.info
  financial_currency = info.get('financialCurrency')
  trading_currency = info.get('currency')

  if not financial_currency or not trading_currency:
    return 1.0

  return get_fx_rate(financial_currency, trading_currency)


def is_foreign_currency_filer(stock: yf.Ticker) -> bool:
  """Check if the statements are in a different currency than the share price.

  Args:
      stock: the stock in question as a yf.Ticker object

  Returns:
      True if the reporting currency differs from the trading currency.
  """
  info = stock.info
  financial_currency = info.get('financialCurrency')
  return bool(financial_currency) and financial_currency != info.get('currency')


def get_country_risk_premium(stock: yf.Ticker) -> float:
  """Get the extra discount rate for a company's country and reporting currency.

  Takes the higher of the two, because Yahoo's country is often the legal
  domicile (e.g. Cayman Islands) while the reporting currency shows where the
  business actually earns its money.

  Args:
      stock: the stock in question as a yf.Ticker object

  Returns:
      The premium as a decimal (0.05 == 5 percentage points).
  """
  info = stock.info
  country = info.get('country')
  currency = info.get('financialCurrency')

  country_premium = COUNTRY_RISK_PREMIUMS.get(country, DEFAULT_FOREIGN_PREMIUM) if country else 0.0
  currency_premium = CURRENCY_RISK_PREMIUMS.get(currency, DEFAULT_FOREIGN_PREMIUM) if currency else 0.0

  return max(country_premium, currency_premium)


def get_fcf_growth_rate(stock: yf.Ticker) -> float:
  """Estimate the starting free cash flow growth rate.

  Uses the compound annual growth rate (CAGR) of free cash flow between the
  oldest and newest available years. Falls back to the analyst earnings growth
  estimate when CAGR can't be computed (e.g. negative FCF in a year).
  The result is clamped between 0 and MAX_GROWTH.

  Args:
      stock: the stock in question as a yf.Ticker object

  Returns:
      A growth rate as a decimal (0.08 == 8%).
  """
  fcf = get_free_cash_flow_history(stock)
  growth = None

  if len(fcf) >= 2:
    newest = fcf.iloc[0]
    oldest = fcf.iloc[-1]
    years = len(fcf) - 1
    if newest > 0 and oldest > 0:
      growth = (newest / oldest) ** (1 / years) - 1

  if growth is None:
    growth = stock.info.get('earningsGrowth')

  if growth is None:
    growth = 0.0

  return min(max(growth, 0.0), MAX_GROWTH)


def get_net_debt(stock: yf.Ticker) -> float:
  """Calculate net debt (total debt - cash) from the latest balance sheet.

  Args:
      stock: the stock in question as a yf.Ticker object

  Returns:
      Net debt as a float. Negative means the company holds net cash.
  """
  balance_sheet = stock.balance_sheet
  total_debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 0.0
  cash = balance_sheet.loc['Cash Cash Equivalents And Short Term Investments'].iloc[0] \
    if 'Cash Cash Equivalents And Short Term Investments' in balance_sheet.index \
    else balance_sheet.loc['Cash And Cash Equivalents'].iloc[0]

  return float(total_debt) - float(cash)


def get_earnings_history(stock: yf.Ticker) -> pd.Series:
  """Get yearly earnings excluding one-time items, most recent year first.

  Uses 'Normalized Income', falling back to 'Net Income' when it isn't reported.

  Args:
      stock: the stock in question as a yf.Ticker object

  Returns:
      A pandas Series of earnings indexed by fiscal year end date (empty if unavailable).
  """
  income_statement = yearly.drop_incomplete_periods(stock.income_stmt)

  for row in ['Normalized Income', 'Net Income']:
    if row in income_statement.index and income_statement.loc[row].notna().any():
      return income_statement.loc[row].dropna()

  return pd.Series(dtype=float)


def get_base_fcf(stock: yf.Ticker) -> float:
  """Get the starting free cash flow for the DCF, robust to one-off years.

  1. Smoothed FCF = the higher of the average of the last FCF_AVERAGE_YEARS
     years and the median of all years, so a one-off cash payment in a recent
     year doesn't drag the base down.
  2. Earnings cap = latest earnings x the median FCF / earnings ratio over all
     years. When cash flow falls because the business is shrinking, earnings
     fall too and the cap pulls the base down.

  Base FCF = min(smoothed FCF, earnings cap). Without usable earnings
  history the plain average is used, which is the conservative choice.

  Args:
      stock: the stock in question as a yf.Ticker object

  Returns:
      The base free cash flow in the reporting currency.
  """
  fcf = get_free_cash_flow_history(stock)
  average_fcf = float(fcf.iloc[:FCF_AVERAGE_YEARS].mean())
  smoothed_fcf = max(average_fcf, float(fcf.median()))

  earnings = get_earnings_history(stock)
  if earnings.empty or earnings.iloc[0] <= 0:
    return average_fcf

  # compare cash flow and earnings for the same fiscal years
  common_years = fcf.index.intersection(earnings[earnings > 0].index)
  if common_years.empty:
    return average_fcf

  conversion = (fcf[common_years] / earnings[common_years]).median()
  earnings_cap = float(earnings.iloc[0] * conversion)

  return min(smoothed_fcf, earnings_cap)


def get_present_value_of_fcf(base_fcf: float, growth: float, discount_rate: float,
                             terminal_growth: float = TERMINAL_GROWTH,
                             years: int = PROJECTION_YEARS) -> float:
  """Present value of projected free cash flows plus a terminal value.

  FCF grows at `growth` for the first half of the projection period, then fades
  linearly to the terminal growth rate. The terminal value uses the Gordon growth model.

  Args:
      base_fcf: the starting free cash flow
      growth: the starting annual growth rate as a decimal
      discount_rate: required annual return as a decimal
      terminal_growth: perpetual growth rate after the projection period
      years: number of years to project

  Returns:
      The total present value of the company's future free cash flow.
  """
  high_growth_years = years // 2

  present_value = 0.0
  cash_flow = base_fcf
  for year in range(1, years + 1):
    if year <= high_growth_years:
      year_growth = growth
    else:
      fade = (year - high_growth_years) / (years - high_growth_years)
      year_growth = growth - (growth - terminal_growth) * fade
    cash_flow *= 1 + year_growth
    present_value += cash_flow / (1 + discount_rate) ** year

  terminal_value = cash_flow * (1 + terminal_growth) / (discount_rate - terminal_growth)
  present_value += terminal_value / (1 + discount_rate) ** years

  return present_value


def get_dcf_intrinsic_value(stock: yf.Ticker, discount_rate: float = DISCOUNT_RATE,
                            terminal_growth: float = TERMINAL_GROWTH,
                            years: int = PROJECTION_YEARS) -> Optional[float]:
  """Estimate intrinsic value per share with a two-stage DCF on free cash flow.

  Stage 1: FCF grows at the estimated growth rate for the first half of the
  projection period, then fades linearly to the terminal growth rate.
  Stage 2: a terminal value using the Gordon growth model.

  Equity Value = PV(projected FCF) + PV(terminal value) - net debt
  Intrinsic Value per Share = Price x Equity Value / Market Cap

  Equity value is converted into the trading currency, and comparing it to the
  market cap (rather than dividing by a share count) keeps ADRs and companies
  with several share classes on the same per share basis as the price.

  Args:
      stock: the stock in question as a yf.Ticker object
      discount_rate: required annual return as a decimal
      terminal_growth: perpetual growth rate after the projection period
      years: number of years to project

  Returns:
      Intrinsic value per share, or None if the company has negative FCF.
  """
  base_fcf = get_base_fcf(stock)

  # A DCF is meaningless when the company burns cash.
  if base_fcf <= 0:
    return None

  growth = get_fcf_growth_rate(stock)
  present_value = get_present_value_of_fcf(base_fcf, growth, discount_rate, terminal_growth, years)

  equity_value = (present_value - get_net_debt(stock)) * get_currency_factor(stock)
  per_share = get_current_price(stock) * equity_value / get_market_cap(stock)

  return round(per_share, 2)


def get_implied_growth(stock: yf.Ticker, discount_rate: float = DISCOUNT_RATE) -> Optional[float]:
  """Reverse DCF: the FCF growth rate the current share price assumes.

  Solves for the starting growth rate that makes the DCF value equal the price,
  using the same model and inputs as get_dcf_intrinsic_value. So a stock is
  overvalued when this is above fcf_growth_used and undervalued when below.
  Compare it to the company's actual growth to judge if the price is realistic.

  Args:
      stock: the stock in question as a yf.Ticker object
      discount_rate: required annual return as a decimal

  Returns:
      A floating point percentage, or None if the company has negative FCF or
      the price implies growth outside IMPLIED_GROWTH_RANGE.
  """
  base_fcf = get_base_fcf(stock)

  if base_fcf <= 0:
    return None

  # value needed from future cash flows (in the reporting currency) to justify the market cap
  target = get_market_cap(stock) / get_currency_factor(stock) + get_net_debt(stock)

  low, high = IMPLIED_GROWTH_RANGE
  if not (get_present_value_of_fcf(base_fcf, low, discount_rate) <= target
          <= get_present_value_of_fcf(base_fcf, high, discount_rate)):
    return None

  # value rises with growth, so bisect until the growth rate is pinned down
  for _ in range(60):
    middle = (low + high) / 2
    if get_present_value_of_fcf(base_fcf, middle, discount_rate) < target:
      low = middle
    else:
      high = middle

  return percent_round(low * 100)


def get_graham_number(stock: yf.Ticker) -> Optional[float]:
  """Calculate Benjamin Graham's fair value ceiling.

  Graham Number = sqrt(22.5 x EPS x Book Value per Share)
  Price below the Graham number suggests the stock is cheap on assets and earnings.
  Tends to undervalue asset-light, high-ROIC companies, so use as a cross-check.

  Args:
      stock: the stock in question as a yf.Ticker object

  Returns:
      The Graham number, or None if EPS or book value is negative, or the company
      reports in a foreign currency (Yahoo mixes currencies for EPS and book value).
  """
  if is_foreign_currency_filer(stock):
    return None

  info = stock.info
  eps = info.get('trailingEps')
  book_value = info.get('bookValue')

  if not eps or not book_value or eps <= 0 or book_value <= 0:
    return None

  return round(math.sqrt(22.5 * eps * book_value), 2)


def get_fcf_yield(stock: yf.Ticker) -> float:
  """Calculate free cash flow yield (FCF / market cap).

  Comparable to an earnings yield; higher means cheaper.
  Compare against the 10 year treasury yield.

  Args:
      stock: the stock in question as a yf.Ticker object

  Returns:
      A floating point percentage.
  """
  fcf = get_free_cash_flow_history(stock).iloc[0] * get_currency_factor(stock)

  return percent_round(fcf / get_market_cap(stock) * 100)


def get_margin_of_safety(intrinsic_value: Optional[float], price: float) -> Optional[float]:
  """Calculate the discount of the current price to intrinsic value.

  Margin of Safety = (Intrinsic Value - Price) / Intrinsic Value
  A positive value means the stock trades below its estimated value.

  Args:
      intrinsic_value: the estimated value per share
      price: the current share price

  Returns:
      A floating point percentage, or None if intrinsic value is not positive.
  """
  if intrinsic_value is None or intrinsic_value <= 0:
    return None

  return percent_round((intrinsic_value - price) / intrinsic_value * 100)


def get_valuation(stock: yf.Ticker) -> dict:
  """Run all valuation metrics for a stock.

  Args:
      stock: the stock in question as a yf.Ticker object

  Returns:
      A dict of valuation metrics ready to merge into the results row.
      FCF based metrics are None for sectors where FCF isn't meaningful.
  """
  info = stock.info
  sector = info.get('sector')
  price = get_current_price(stock)

  # foreign companies are discounted at a higher rate for country and currency risk
  discount_rate = DISCOUNT_RATE + get_country_risk_premium(stock)

  fcf_reliable = sector not in UNRELIABLE_FCF_SECTORS
  intrinsic_value = get_dcf_intrinsic_value(stock, discount_rate=discount_rate) if fcf_reliable else None
  margin_of_safety = get_margin_of_safety(intrinsic_value, price)

  return {
    "sector": sector,
    "country": info.get('country'),
    "financial_currency": info.get('financialCurrency'),
    "discount_rate": percent_round(discount_rate * 100),
    "price": price,
    "fcf_growth_used": percent_round(get_fcf_growth_rate(stock) * 100),
    "implied_growth": get_implied_growth(stock, discount_rate=discount_rate) if fcf_reliable else None,
    "dcf_value": intrinsic_value,
    "margin_of_safety": margin_of_safety,
    "graham_number": get_graham_number(stock),
    "fcf_yield": get_fcf_yield(stock) if fcf_reliable else None,
    "undervalued": margin_of_safety is not None and margin_of_safety >= MARGIN_OF_SAFETY_TARGET
  }
