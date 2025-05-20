import requests
from bs4 import BeautifulSoup
import sqlite3

# URLs to scrape from ask.co.ke
urls = [
    "https://ask.co.ke/#",
    "https://ask.co.ke/leadership/",
    "https://ask.co.ke/career/",
    "https://ask.co.ke/ask-membership/",
    "https://ask.co.ke/register-as-an-exhibitor/",
    "https://ask.co.ke/contact-us/"
]

def scrape_page(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract title and content (target main content area)
        title = soup.find("title").get_text(strip=True) if soup.find("title") else url
        content = soup.find("div", class_="entry-content") or soup.find("body")
        text = content.get_text(separator=" ", strip=True) if content else ""

        return {"url": url, "title": title, "content": text}
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def save_to_db(pages):
    conn = sqlite3.connect("ask_content.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pages (
            url TEXT PRIMARY KEY,
            title TEXT,
            content TEXT
        )
    """)
    for page in pages:
        if page:
            cursor.execute(
                "INSERT OR REPLACE INTO pages (url, title, content) VALUES (?, ?, ?)",
                (page["url"], page["title"], page["content"])
            )
    conn.commit()
    conn.close()

def main():
    pages = [scrape_page(url) for url in urls]
    save_to_db(pages)
    print("Scraping complete. Content saved to ask_content.db")

if __name__ == "__main__":
    main()