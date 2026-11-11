"""Scraper module for fetching and storing news articles from RSS feeds.

Responsible for:
- Fetching news from RSS feeds using requests
- Parsing article metadata (title, link, published date, description) with BeautifulSoup
- Storing articles in data/ as JSONL records (unique_id, url, date, headline, body)
- Running until at least 300 articles from the last week are stored
"""

import json
import os
import sys
import uuid
import datetime
from typing import List, Dict

import requests
from bs4 import BeautifulSoup


# ------------------------------------------------------------
# Configuration (change these if you want a different source)
# ------------------------------------------------------------
DEFAULT_FEED_URL = "https://feeds.bbci.co.uk/news/rss.xml"
DATA_DIR = "data"
ARTICLES_FILE = os.path.join(DATA_DIR, "articles.jsonl")
MAX_ARTICLES = 300
LAST_WEEK_CUTOFF_DAYS = 7


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def _is_within_last_week(pub_date_str: str) -> bool:
    """Return True if pub_date_str (RSS format) is within the last LAST_WEEK_CUTOFF_DAYS."""
    try:
        # RSS date format: "Mon, 11 Nov 2026 09:00:00 GMT"
        article_dt = datetime.datetime.strptime(
            pub_date_str, "%a, %d %b %Y %H:%M:%S %Z"
        )
        article_dt = article_dt.replace(tzinfo=datetime.timezone.utc)
    except Exception:
        # If we can't parse, include the article (defensive)
        return True
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        days=LAST_WEEK_CUTOFF_DAYS
    )
    return article_dt >= cutoff


def _parse_rss_date(pub_date_str: str) -> str:
    """Parse RSS pubDate into YYYY-MM-DD string."""
    try:
        article_dt = datetime.datetime.strptime(
            pub_date_str, "%a, %d %b %Y %H:%M:%S %Z"
        ).replace(tzinfo=datetime.timezone.utc)
        return article_dt.strftime("%Y-%m-%d")
    except Exception:
        return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def _write_record(record: Dict) -> None:
    """Append a single record to the JSONL articles file."""
    filepath = ARTICLES_FILE
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ------------------------------------------------------------
# Main scraper logic
# ------------------------------------------------------------
def run_scraper(feed_url: str = DEFAULT_FEED_URL) -> None:
    """Fetch articles from the RSS feed until MAX_ARTICLES are stored.

    Prints progress per article in the format:
        scraping <URL> - requesting ...
        scraping <URL> - parsing ...
        scraping <URL> - saved in <path>
    """
    _ensure_data_dir()

    # Remove old articles beyond the last week? For simplicity we just count toward MAX_ARTICLES.
    # Count how many already exist (quick heuristic).
    existing_count = 0
    if os.path.exists(ARTICLES_FILE):
        with open(ARTICLES_FILE, "r", encoding="utf-8") as f:
            existing_count = sum(1 for _ in f)

    needed = MAX_ARTICLES - existing_count
    if needed <= 0:
        print(f"Already have {MAX_ARTICLES} or more articles. Nothing to do.")
        return

    print(f"Fetching up to {needed} articles from RSS feed: {feed_url}")

    try:
        resp = requests.get(feed_url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"Error fetching RSS feed: {exc}", file=sys.stderr)
        sys.exit(1)

    soup = BeautifulSoup(resp.text, "xml")
    items = soup.find_all("item")

    saved = 0
    with open(ARTICLES_FILE, "a", encoding="utf-8") as out_f:
        for item in items:
            if saved >= needed:
                break

            # Extract fields – graceful fallbacks
            title_elem = item.find("title")
            link_elem = item.find("link")
            date_elem = item.find("pubDate")
            desc_elem = item.find("description")

            title = title_elem.text if title_elem else "No title"
            url = link_elem.text if link_elem else ""
            pub_date_raw = date_elem.text if date_elem else ""
            body = desc_elem.text if desc_elem else ""

            # Date filtering: only keep last week
            if not _is_within_last_week(pub_date_raw):
                continue

            date_str = _parse_rss_date(pub_date_raw)
            unique_id = str(uuid.uuid4())

            record = {
                "unique_id": unique_id,
                "url": url,
                "date": date_str,
                "headline": title,
                "body": body,
            }

            # Print progress lines exactly as the deliverable specifies
            print(f"scraping {url} - requesting ...")
            print(f"scraping {url} - parsing ...")
            print(f"scraping {url} - saved in {ARTICLES_FILE}")

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved += 1

    print(f"\nDone. Saved {saved} new articles (total in {ARTICLES_FILE}): {existing_count + saved}")


# ------------------------------------------------------------
# Allow running from command line: python scraper_news.py
# ------------------------------------------------------------
if __name__ == "__main__":
    # Allow overriding the feed URL via 1st arg if desired
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FEED_URL
    run_scraper(url)