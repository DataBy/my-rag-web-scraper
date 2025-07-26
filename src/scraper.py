import requests
from bs4 import BeautifulSoup

def scrape_website(url):
    try:
        # Send HTTP GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad responses (4xx, 5xx)

        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove unwanted tags like scripts, styles, headers, etc.
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()  # Completely remove the tag and its contents

        # Extract only the clean text, separated by spaces
        text = soup.get_text(separator=" ", strip=True)

        return text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None
