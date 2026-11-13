'''This is a module for ranking undervalued stocks by how safe a buy they are
   and writing a short analysis of each one for the summary email.

   A stock with a huge margin of safety isn't necessarily a safe buy. The deepest
   discounts are often penny stocks, high risk countries or businesses the market
   expects to shrink. So the safety score favours a strong balance sheet, high
   returns on capital, low country risk and a price that assumes little growth.
'''
import math
from typing import List, Optional

import pandas as pd

from valuation import DISCOUNT_RATE, MARGIN_OF_SAFETY_TARGET


# How many of the safest buys to list in the summary email.
SAFEST_BUYS_COUNT = 15
# Stocks scoring below this are left out of the email, however undervalued they look.
MIN_SAFETY_SCORE = 50

# Each metric is ranked against the other undervalued stocks and the ranks are
# averaged with these weights. True means a higher value is safer.
SAFETY_METRICS = {
  'country_premium': (False, 2),
  'net_debt/fcf': (False, 2),
  'debt_equity': (False, 1),
  'roic': (True, 1),
  'fcf_earnings': (True, 1),
  'implied_growth': (False, 1),
  'margin_of_safety': (True, 1),
}

# Shares below this price tend to be thinly traded and volatile.
PENNY_STOCK_PRICE = 5
PENNY_STOCK_PENALTY = 15
# Trading below net cash often means the cash isn't really available to shareholders.
BELOW_NET_CASH_PENALTY = 10
# A discount this deep usually means the market sees a problem the model doesn't.
SUSPICIOUS_MARGIN_OF_SAFETY = 70


def get_safety_scores(undervalued_df: pd.DataFrame) -> pd.Series:
  """Score each undervalued stock from 0 (riskiest) to 100 (safest).

  Args:
      undervalued_df: stocks that met the margin of safety target

  Returns:
      A Series of scores aligned with undervalued_df's index.
  """
  df = undervalued_df.copy()
  df['country_premium'] = df['discount_rate'] - DISCOUNT_RATE * 100
  # no implied growth below net cash means the price assumes the business is worth nothing
  df['implied_growth'] = df['implied_growth'].mask(df['below_net_cash'].eq(True), -100)

  weighted_ranks = pd.Series(0.0, index=df.index)
  total_weight = 0
  for column, (higher_is_safer, weight) in SAFETY_METRICS.items():
    if column not in df.columns:
      continue
    # missing values get a middling rank rather than a best or worst one
    ranks = df[column].rank(pct=True, ascending=higher_is_safer).fillna(0.5)
    weighted_ranks += ranks * weight
    total_weight += weight

  scores = weighted_ranks / total_weight * 100
  scores -= df['price'].lt(PENNY_STOCK_PRICE) * PENNY_STOCK_PENALTY
  scores -= df['below_net_cash'].eq(True) * BELOW_NET_CASH_PENALTY
  return scores.clip(lower=0).round(1)


def is_missing(value: Optional[float]) -> bool:
  """Check if a metric is missing (None or NaN)."""
  return value is None or (isinstance(value, float) and math.isnan(value))


def get_strengths(row: pd.Series) -> List[str]:
  """List what makes the stock a safer buy.

  Args:
      row: one stock's results row

  Returns:
      A list of short sentences.
  """
  strengths = []
  country_premium = row['discount_rate'] - DISCOUNT_RATE * 100

  if row['net_debt/fcf'] < 0:
    strengths.append("Holds more cash than debt")
  elif row['net_debt/fcf'] < 2:
    strengths.append(f"Low net debt, {row['net_debt/fcf']:.1f}x free cash flow")
  if row['debt_equity'] < 30:
    strengths.append(f"Low debt/equity of {row['debt_equity']:.0f}%")
  if row['roic'] >= 25:
    strengths.append(f"High return on invested capital of {row['roic']:.0f}%")
  if row['fcf_earnings'] >= 100:
    strengths.append(f"Earnings fully backed by cash (FCF/earnings {row['fcf_earnings']:.0f}%)")
  if country_premium <= 0:
    strengths.append("US company, no country risk premium")
  elif country_premium <= 1:
    strengths.append(f"Developed market ({row['country']}), low country risk")
  if not is_missing(row['implied_growth']) and row['implied_growth'] < 0:
    strengths.append(f"Price assumes FCF shrinks {-row['implied_growth']:.1f}% a year "
                     f"vs {row['fcf_growth_used']:.0f}% growth used in the DCF")
  elif not is_missing(row['implied_growth']) and row['implied_growth'] <= 5:
    strengths.append(f"Price assumes only {row['implied_growth']:.1f}% FCF growth "
                     f"vs {row['fcf_growth_used']:.0f}% used in the DCF")
  if not is_missing(row.get('fcf_yield')) and row['fcf_yield'] >= 8:
    strengths.append(f"FCF yield of {row['fcf_yield']:.1f}%")

  return strengths


def get_risks(row: pd.Series) -> List[str]:
  """List what could make the stock a worse buy than the numbers suggest.

  Args:
      row: one stock's results row

  Returns:
      A list of short sentences.
  """
  risks = []
  country_premium = row['discount_rate'] - DISCOUNT_RATE * 100

  if country_premium >= 2:
    risks.append(f"{row['country']} adds a {country_premium:.1f}% country risk premium")
  if row['price'] < PENNY_STOCK_PRICE:
    risks.append(f"Penny stock at ${row['price']:.2f}, often thinly traded and volatile")
  if row['below_net_cash']:
    risks.append("Trades below net cash, check the cash is really available to shareholders")
  if row['margin_of_safety'] >= SUSPICIOUS_MARGIN_OF_SAFETY:
    risks.append(f"Discount of {row['margin_of_safety']:.0f}% is unusually deep, "
                 "the market may see a problem the model doesn't")
  elif row['margin_of_safety'] < MARGIN_OF_SAFETY_TARGET + 5:
    risks.append(f"Margin of safety of {row['margin_of_safety']:.0f}% is close to the "
                 f"{MARGIN_OF_SAFETY_TARGET}% target")
  if row['fcf_growth_used'] <= 0:
    risks.append("Free cash flow hasn't grown, the DCF assumes no growth")
  if row['net_debt/fcf'] >= 3:
    risks.append(f"Net debt is {row['net_debt/fcf']:.1f}x free cash flow")
  if row['debt_equity'] >= 50:
    risks.append(f"Debt/equity of {row['debt_equity']:.0f}%")
  if row['earnings_growth'] > 50:
    risks.append(f"Earnings growth of {row['earnings_growth']:.0f}% may be a rebound that won't repeat")

  return risks


def get_safest_undervalued(undervalued_df: pd.DataFrame) -> pd.DataFrame:
  """Pick the undervalued stocks with the highest safety scores.

  Args:
      undervalued_df: stocks that met the margin of safety target

  Returns:
      Up to SAFEST_BUYS_COUNT stocks scoring at least MIN_SAFETY_SCORE, with a safety_score column,
      most undervalued (largest margin of safety) first.
  """
  df = undervalued_df.assign(safety_score=get_safety_scores(undervalued_df))
  df = df[df['safety_score'] >= MIN_SAFETY_SCORE]
  df = df.sort_values(by=['safety_score', 'margin_of_safety'], ascending=[False, False])
  return df.head(SAFEST_BUYS_COUNT).sort_values(by='margin_of_safety', ascending=False)


def build_safest_buys_analysis(safest_df: pd.DataFrame) -> List[str]:
  """Write a short analysis of each of the safest undervalued stocks.

  Args:
      safest_df: stocks returned by get_safest_undervalued

  Returns:
      The analysis as a list of lines.
  """
  lines = []
  for position, (_, row) in enumerate(safest_df.iterrows(), start=1):
    lines.append(f"{position}. {row['ticker']} ({row['sector']}, {row['country']}) "
                 f"safety score {row['safety_score']:.0f}/100")
    lines.append(f"   Price ${row['price']:.2f} vs DCF value ${row['dcf_value']:.2f} "
                 f"({row['margin_of_safety']:.0f}% margin of safety)")
    lines += [f"   + {strength}" for strength in get_strengths(row)]
    lines += [f"   - {risk}" for risk in get_risks(row)]
    lines.append("")

  return lines
