'''This is a module for emailing a summary when a scrape finishes.

   Sends through Gmail's SMTP server. Gmail requires an app password
   (not your normal password): https://myaccount.google.com/apppasswords

   Set these environment variables before running main.py:
     GMAIL_ADDRESS       the Gmail account sending the email
     GMAIL_APP_PASSWORD  the 16 character app password for that account
     NOTIFY_EMAIL        (optional) recipient, defaults to NOTIFY_EMAIL_DEFAULT
'''
import logging
import os
import smtplib
from email.message import EmailMessage
from typing import Optional

import pandas as pd


logger = logging.getLogger(__name__)

SMTP_HOST = 'smtp.gmail.com'
SMTP_PORT = 587
NOTIFY_EMAIL_DEFAULT = 'stanutz.stanutz@gmail.com'

SUMMARY_COLUMNS = ['ticker', 'price', 'dcf_value', 'margin_of_safety', 'fcf_growth_used', 'implied_growth', 'undervalued']


def build_summary_email(tickers_processed: int, results_count: int,
                        filtered_df: Optional[pd.DataFrame], timestamp: str) -> EmailMessage:
  """Build the scrape complete email.

  Args:
      tickers_processed: how many tickers the run attempted
      results_count: how many tickers returned metrics
      filtered_df: the stocks that passed the screen, or None if there were no results
      timestamp: the run timestamp used in the result file names

  Returns:
      An EmailMessage without sender/recipient set.
  """
  passed = 0 if filtered_df is None else len(filtered_df)
  undervalued = 0
  if filtered_df is not None and 'undervalued' in filtered_df.columns:
    undervalued = int(filtered_df['undervalued'].sum())

  lines = [
    f"Stock scrape {timestamp} is complete.",
    "",
    f"Tickers processed: {tickers_processed}",
    f"Tickers with metrics: {results_count}",
    f"Passed quality screen: {passed}",
    f"Undervalued (margin of safety target met): {undervalued}",
    "",
  ]

  if results_count == 0:
    lines.append("No results were returned. Check getrich.log, this is often caused by rate limiting.")
  elif passed:
    columns = [column for column in SUMMARY_COLUMNS if column in filtered_df.columns]
    lines.append("Stocks passing the screen:")
    lines.append(filtered_df[columns].to_string(index=False))

  lines += ["", f"Results are saved under results/ with timestamp {timestamp}."]

  message = EmailMessage()
  message['Subject'] = f"Stock scrape complete: {passed} passed, {undervalued} undervalued"
  message.set_content("\n".join(lines))
  return message


def build_failure_email(error_traceback: str, timestamp: str) -> EmailMessage:
  """Build the scrape failed email.

  Args:
      error_traceback: the formatted traceback of the crash
      timestamp: when the failure happened

  Returns:
      An EmailMessage without sender/recipient set.
  """
  error_line = error_traceback.strip().splitlines()[-1]

  lines = [
    f"Stock scrape failed at {timestamp}.",
    "",
    f"Error: {error_line}",
    "",
    "Results saved before the crash are in stock_analysis_backup.parquet (saved after every batch).",
    "See getrich.log for details.",
    "",
    "Traceback:",
    error_traceback,
  ]

  message = EmailMessage()
  message['Subject'] = f"Stock scrape FAILED: {error_line[:100]}"
  message.set_content("\n".join(lines))
  return message


def send_email(message: EmailMessage, attachment_path: Optional[str] = None) -> bool:
  """Send an email through Gmail, optionally attaching a PDF.

  Never raises, so a mail problem can't crash the end of a long scrape.

  Args:
      message: the email to send
      attachment_path: optional path of a PDF to attach

  Returns:
      True if the email was sent, False otherwise.
  """
  sender = os.environ.get('GMAIL_ADDRESS')
  password = os.environ.get('GMAIL_APP_PASSWORD')

  if not sender or not password:
    logger.warning("GMAIL_ADDRESS or GMAIL_APP_PASSWORD not set, skipping email notification")
    print("GMAIL_ADDRESS or GMAIL_APP_PASSWORD not set, skipping email notification")
    return False

  message['From'] = sender
  message['To'] = os.environ.get('NOTIFY_EMAIL', NOTIFY_EMAIL_DEFAULT)

  if attachment_path and os.path.exists(attachment_path):
    with open(attachment_path, 'rb') as attachment:
      message.add_attachment(attachment.read(), maintype='application', subtype='pdf',
                             filename=os.path.basename(attachment_path))

  try:
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
      server.starttls()
      server.login(sender, password)
      server.send_message(message)
  except (smtplib.SMTPException, OSError) as e:
    logger.error(f"Error sending email notification: {str(e)}")
    print(f"Error sending email notification: {str(e)}")
    return False

  logger.info(f"Email notification sent to {message['To']}")
  print(f"Email notification sent to {message['To']}")
  return True
