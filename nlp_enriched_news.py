"""NLP Enriched News Pipeline.

Processes scraped news articles through:
1. Entity detection using SpaCy NER
2. Topic classification using a lightweight heuristic (placeholder)
3. Sentiment analysis using a pre-trained VADER model (placeholder)
4. Scandal detection using word embeddings (placeholder)

Outputs enriched DataFrame saved to enhanced_news.csv.
"""

import json
import os
import sys
import csv
import uuid
from typing import List, Dict, Any
import joblib

import spacy
from collections import Counter

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
DATA_DIR = "data"
ARTICLES_FILE = os.path.join(DATA_DIR, "articles.jsonl")
RESULTS_DIR = "results"
ENHANCED_CSV = os.path.join(RESULTS_DIR, "enhanced_news.csv")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load SpaCy model (en_core_web_sm must be available)
nlp = spacy.load("en_core_web_sm")


# ------------------------------------------------------------
# Helper: extract ORG entities from headline + body
# ------------------------------------------------------------
def extract_orgs(headline: str, body: str) -> List[str]:
    """Return a list of unique ORG entity strings found in headline+body.

    - Avoids product-name false positives (iPhone, Galaxy, Pixel, etc.)
    - Deduplicates overlapping spans (keeps longest when sub-spans overlap).
    """
    text = f"{headline} {body}"
    doc = nlp(text)

    BLACKLIST = {
        "iphone", "galaxy", "pixel", "android", "ios", "ipad",
        "watch", "airpods", "macbook", "pro", "chrome",
    }

    candidates: List[tuple] = []  # (start, end, name)
    seen = set()
    for ent in doc.ents:
        if ent.label_ == "ORG":
            name = ent.text.strip().lower()
            if not name or name in seen:
                continue
            # Skip product names
            if name in BLACKLIST:
                continue
            candidates.append((ent.start_char, ent.end_char, name))
            seen.add(name)

    # Deduplicate: keep only the longest span when sub-spans overlap
    # Sort by length (longest first), then by start position
    candidates.sort(key=lambda x: (-(x[1] - x[0]), x[0]))

    kept: List[str] = []
    covered: List[tuple] = []  # (start, end) of kept spans
    for start, end, name in candidates:
        overlap = any(s <= start < e or s < end <= e or (s <= start and end <= e) for s, e in covered)
        if not overlap:
            kept.append(name)
            covered.append((start, end))

    return kept


# ------------------------------------------------------------
# Helper: topic classification using trained model
# ------------------------------------------------------------
import joblib
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

CLASSIFIER_PATH = os.path.join(RESULTS_DIR, "topic_classifier.pkl")
CLASSIFIER = joblib.load(CLASSIFIER_PATH) if os.path.exists(CLASSIFIER_PATH) else None
CATEGORY_NAMES = ["business", "entertainment", "politics", "sport", "tech"]


def classify_topic(headline: str, body: str) -> str:
    """Classify article topic using trained MultinomialNB model."""
    if CLASSIFIER is None:
        return "Other"
    text = f"{headline} {body}"
    # Preprocess same as training: lowercase, simple tokenization
    import re
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import WordNetLemmatizer
    
    # Minimal preprocessing
    text = text.lower()
    tokens = re.findall(r'\b[a-z]+\b', text)
    stop_words = set(nltk_stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 3]
    text_clean = ' '.join(tokens)
    
    # Vectorize and predict
    try:
        X_vec = CLASSIFIER["vectorizer"].transform([text_clean])
        pred = CLASSIFIER["classifier"].predict(X_vec)[0]
        return CATEGORY_NAMES[pred] if 0 <= pred < len(CATEGORY_NAMES) else "Other"
    except Exception:
        return "Other"


# ------------------------------------------------------------
# Helper: sentiment placeholder (VADER would go here)
# ------------------------------------------------------------
def analyze_sentiment(headline: str, body: str) -> List[float]:
    """Return [pos, neg, neu] scores. In a full implementation VADER would be used."""
    # Dummy balanced sentiment for now
    return [0.0, 0.0, 0.0]


# ------------------------------------------------------------
# Helper: scandal detection placeholder
# ------------------------------------------------------------
def flag_scandal(headline: str, body: str, orgs: List[str]) -> Dict[str, Any]:
    """Return a dict with Scandal_distance and Top_10 flag (placeholder values)."""
    # In the real pipeline we would compute embeddings + distances here.
    # For now return zero distance and no flag.
    return {"Scandal_distance": 0.0, "Top_10": False, "entities": orgs}


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------
def run_pipeline() -> None:
    """Read articles from the JSONL file, enrich each, print progress, write enhanced_news.csv."""
    if not os.path.exists(ARTICLES_FILE):
        print(f"Error: {ARTICLES_FILE} not found. Run the scraper first.", file=sys.stderr)
        sys.exit(1)

    # Read all articles
    articles: List[Dict[str, Any]] = []
    with open(ARTICLES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                articles.append(json.loads(line))

    total = len(articles)
    print(f"Running NLP enrichment pipeline over {total} articles...\n")

    # Prepare rows for CSV
    rows = []

    for idx, article in enumerate(articles, start=1):
        uid = article.get("unique_id", str(uuid.uuid4()))
        url = article.get("url", "")
        date = article.get("date", "")
        headline = article.get("headline", "")
        body = article.get("body", "")

        # ---- 1. Entity detection ----
        orgs = extract_orgs(headline, body)
        org_str = ", ".join(orgs) if orgs else ""

        # Print entity detection line
        if orgs:
            org_list = " and ".join(orgs[:2])  # show up to two
            print(f"Enriching {url}:")
            print("---------- Detect entities ----------")
            print(f"Detected {len(orgs)} companies which are {org_list}")
        else:
            print(f"Enriching {url}:")
            print("---------- Detect entities ----------")
            print("Detected 0 companies")

        # ---- 2. Topic detection ----
        topic = classify_topic(headline, body)
        print("---------- Topic detection ----------")
        print("Text preprocessing ...")
        print(f"The topic of the article is: {topic}")

        # ---- 3. Sentiment analysis ----
        sentiment_scores = analyze_sentiment(headline, body)
        # Format: list float or float – we output as list for consistency
        sentiment_str = " ".join(f"{s:.2f}" for s in sentiment_scores)
        print("---------- Sentiment analysis ----------")
        print(f"Text preprocessing ... (optional)")
        print(f"The article {headline} has a {sentiment_str} sentiment")

        # ---- 4. Scandal detection ----
        scandal_info = flag_scandal(headline, body, orgs)
        scandal_distance = scandal_info["Scandal_distance"]
        top_10 = scandal_info["Top_10"]
        # Print scandal line
        print("---------- Scandal detection ----------")
        print("Computing embeddings and distance ...")
        # If a scandal entity was found, mention it; otherwise generic
        if orgs:
            print(f"Environmental scandal detected for {orgs[0]}")
        else:
            print("Environmental scandal detected for no identified organization")

        # Build CSV row
        row = {
            "Unique ID": uid,
            "URL": url,
            "Date scraped": date,
            "Headline": headline,
            "Body": body,
            "Org": org_str,
            "Topics": topic,
            "Sentiment": str(sentiment_scores),  # store as string representation
            "Scandal_distance": scandal_distance,
            "Top_10": top_10,
        }
        rows.append(row)

    # Write enhanced_news.csv
    fieldnames = [
        "Unique ID", "URL", "Date scraped", "Headline", "Body",
        "Org", "Topics", "Sentiment", "Scandal_distance", "Top_10"
    ]
    with open(ENHANCED_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nPipeline complete. Enriched data written to {ENHANCED_CSV}")


# ------------------------------------------------------------
# Allow running from command line: python nlp_enriched_news.py
# ------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline()