"""Scraper module for fetching and storing news articles from RSS feeds.

Responsible for:
- Fetching news from RSS feeds using requests
- Parsing article metadata and full article body with BeautifulSoup
- Storing articles in data/ as JSONL records (unique_id, url, date, headline, body)
- Running until at least 300 articles from the last week are stored
"""

import json
import os
import sys
import uuid
import datetime
import time
from typing import Dict, List, Set

import requests
from bs4 import BeautifulSoup


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
DEFAULT_FEED_URLS = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "https://feeds.bbci.co.uk/news/politics/rss.xml",
    "https://feeds.bbci.co.uk/news/health/rss.xml",
    "https://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml",
    "https://feeds.bbci.co.uk/news/uk/rss.xml",
    "https://feeds.bbci.co.uk/news/england/rss.xml",
    "https://feeds.bbci.co.uk/news/northern_ireland/rss.xml",
    "https://feeds.bbci.co.uk/news/scotland/rss.xml",
    "https://feeds.bbci.co.uk/news/wales/rss.xml",
    "https://feeds.bbci.co.uk/news/world/africa/rss.xml",
    "https://feeds.bbci.co.uk/news/world/asia/rss.xml",
    "https://feeds.bbci.co.uk/news/world/europe/rss.xml",
    "https://feeds.bbci.co.uk/news/world/latin_america/rss.xml",
    "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml",
    "https://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml",
    "https://feeds.bbci.co.uk/news/education/rss.xml",
    "https://feeds.bbci.co.uk/news/in_pictures/rss.xml",
    "https://feeds.bbci.co.uk/news/magazine/rss.xml",
    "https://feeds.bbci.co.uk/news/topics/c008ql15vdxt/rss.xml",
    "https://feeds.bbci.co.uk/news/topics/c2vdnvdg6xnt/rss.xml",
    "https://www.theguardian.com/world/rss",
    "https://www.theguardian.com/uk-news/rss",
    "https://www.theguardian.com/business/rss",
    "https://www.theguardian.com/technology/rss",
    "https://www.theguardian.com/environment/rss",
    "https://www.theguardian.com/politics/rss",
    "https://www.theguardian.com/science/rss",
    "https://www.theguardian.com/society/rss",
]
DATA_DIR = "data"
ARTICLES_FILE = os.path.join(DATA_DIR, "articles.jsonl")
MAX_ARTICLES = 300
LAST_WEEK_CUTOFF_DAYS = 7
HEADERS = {"User-Agent": "nlp-scraper/1.0 (+educational project)"}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def _clean_text(text: str) -> str:
    return " ".join(text.split()).strip()


def _parse_rss_datetime(pub_date_str: str):
    try:
        dt = datetime.datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %Z")
        return dt.replace(tzinfo=datetime.timezone.utc)
    except Exception:
        return None


def _is_within_last_week(pub_date_str: str) -> bool:
    article_dt = _parse_rss_datetime(pub_date_str)
    if article_dt is None:
        return False
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        days=LAST_WEEK_CUTOFF_DAYS
    )
    return article_dt >= cutoff


def _parse_rss_date(pub_date_str: str) -> str:
    dt = _parse_rss_datetime(pub_date_str)
    if dt is None:
        return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    return dt.strftime("%Y-%m-%d")


def _load_existing_urls() -> Set[str]:
    urls: Set[str] = set()
    if not os.path.exists(ARTICLES_FILE):
        return urls
    with open(ARTICLES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                url = rec.get("url", "")
                if url:
                    urls.add(url)
            except json.JSONDecodeError:
                continue
    return urls


def _extract_article_body(url: str, fallback: str = "") -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException:
        return _clean_text(fallback)

    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    candidates = []
    selectors = [
        "article",
        "main",
        '[data-component="text-block"]',
        '[data-component="text-block"] p',
        "article p",
        "main p",
    ]
    # Prefer text blocks
    blocks = soup.select('[data-component="text-block"]')
    if blocks:
        texts = [_clean_text(b.get_text(" ", strip=True)) for b in blocks]
        texts = [t for t in texts if t]
        if texts:
            return " ".join(texts)

    for sel in ["article p", "main p"]:
        ps = soup.select(sel)
        texts = [_clean_text(p.get_text(" ", strip=True)) for p in ps]
        texts = [t for t in texts if len(t) > 40]
        if texts:
            return " ".join(texts)

    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        return _clean_text(meta["content"])
    return _clean_text(fallback)


# ------------------------------------------------------------
# Main scraper logic
# ------------------------------------------------------------
def run_scraper(feed_urls: List[str] = None) -> None:
    """Fetch articles from RSS feeds until MAX_ARTICLES are stored."""
    if feed_urls is None:
        feed_urls = DEFAULT_FEED_URLS

    _ensure_data_dir()
    seen_urls = _load_existing_urls()
    existing_count = len(seen_urls)
    needed = MAX_ARTICLES - existing_count
    if needed <= 0:
        print(f"Already have {MAX_ARTICLES} or more articles. Nothing to do.")
        return

    print(f"Fetching up to {needed} new articles from {len(feed_urls)} feeds...")

    saved = 0
    with open(ARTICLES_FILE, "a", encoding="utf-8") as out_f:
        for feed_url in feed_urls:
            if saved >= needed:
                break
            try:
                feed_resp = requests.get(feed_url, headers=HEADERS, timeout=30)
                feed_resp.raise_for_status()
            except requests.RequestException as exc:
                print(f"Error fetching RSS feed {feed_url}: {exc}", file=sys.stderr)
                continue

            soup = BeautifulSoup(feed_resp.text, "xml")
            items = soup.find_all("item")
            for item in items:
                if saved >= needed:
                    break

                title_elem = item.find("title")
                link_elem = item.find("link")
                date_elem = item.find("pubDate")
                desc_elem = item.find("description")

                title = _clean_text(title_elem.get_text()) if title_elem else "No title"
                url = _clean_text(link_elem.get_text()) if link_elem else ""
                pub_date_raw = _clean_text(date_elem.get_text()) if date_elem else ""
                desc = _clean_text(desc_elem.get_text()) if desc_elem else ""

                if not url or url in seen_urls:
                    continue
                if not _is_within_last_week(pub_date_raw):
                    continue

                print(f"scraping {url} - requesting ...")
                body = _extract_article_body(url, fallback=desc)
                print(f"scraping {url} - parsing ...")

                record = {
                    "unique_id": str(uuid.uuid4()),
                    "url": url,
                    "date": _parse_rss_date(pub_date_raw),
                    "headline": title,
                    "body": body,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                seen_urls.add(url)
                saved += 1
                print(f"scraping {url} - saved in {ARTICLES_FILE}")
                time.sleep(0.2)

    print(f"\nDone. Saved {saved} new articles (total ~ {existing_count + saved}).")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_scraper(sys.argv[1:])
    else:
        run_scraper()