import sqlite3
from datetime import datetime
import re


REPLACEMENTS = {
  " ": "_",
  ":": "-",
  ".": "-"
}


def create_database():
    try:
        # Create unique db string so we don't loose data
        curr = datetime.today()  
        date_string = parse_date(str(curr), REPLACEMENTS)
        title = f'get-rich-slow_{date_string}.db'
        # Connect to database (creates if it doesn't exist)
        with sqlite3.connect(title) as conn:
            cursor = conn.cursor()
            
            # Create users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    email TEXT NOT NULL UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            print("Database and tables created successfully!")
      
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None
    
    return title

def parse_date(text, replacements):
    # Create a regex pattern from the dictionary keys
    pattern = re.compile("|".join(map(re.escape, replacements.keys())))
    
    # Function to perform the replacement
    def replace_match(match):
        return replacements[match.group(0)]
    
    # Use re.sub() with the pattern and replacement function
    return pattern.sub(replace_match, text)