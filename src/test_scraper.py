import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.scraper import scrape_website


url = "https://www.wikipedia.org/"
text = scrape_website(url)

if text:
    print(text[:1000])  # Print first 1000 characters
