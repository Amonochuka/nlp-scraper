# NLP-Enriched News Intelligence Platform

NLP-enriched News Intelligence platform that scrapes news articles, detects mentioned organizations, classifies topic, scores sentiment, and flags articles most likely related to environmental scandals.

## Project Structure

- `scraper_news.py` - Scrapes recent articles from multiple RSS feeds (BBC and The Guardian), fetches article pages for full body text, de-duplicates by URL, and stores records in JSONL.
- `nlp_enriched_news.py` - NLP enrichment pipeline: ORG entity detection, topic classification, sentiment analysis, and scandal detection.
- `results/training_model.py` - Trains the BBC News topic classifier and generates learning curves.
- `data/articles.jsonl` - Scraped articles (`unique_id`, `url`, `date`, `headline`, `body`), last 7 days, 300+ articles.
- `results/topic_classifier.pkl` - Trained topic classifier (TF-IDF + MultinomialNB).
- `results/learning_curves.png` - Training/validation learning curves.
- `results/enhanced_news.csv` - Enriched output for all scraped articles.

## Deliverables

- `python3 scraper_news.py` scrapes and stores ≥300 recent articles with per-URL progress (`requesting/parsing/saved`).
- `python3 nlp_enriched_news.py` enriches stored articles and writes `results/enhanced_news.csv`.
- `enhanced_news.csv` columns: `Unique ID`, `URL`, `Date scraped`, `Headline`, `Body`, `Org`, `Topics`, `Sentiment`, `Scandal_distance`, `Top_10`.

## Scraper Details

- **Sources**: Multiple RSS feeds from BBC News and The Guardian sections to collect enough unique recent articles.
- **Storage**: `data/articles.jsonl` (one JSON record per line).
- **Filtering**: Keeps articles within the last 7 days and skips duplicate URLs.
- **Body extraction**: Fetches each article URL and extracts main text (with fallback to RSS description) so records include meaningful `body` content.

## NLP Pipeline

### 1. Entity Detection
- **Model**: SpaCy `en_core_web_sm` NER
- **Target**: `ORG` entities from headline + body
- **Post-processing**: blacklist common product-name false positives (e.g. iPhone, Galaxy, Pixel, Android) and remove overlapping duplicate spans (keep longest)

### 2. Topic Detection
- **Dataset**: BBC News dataset (5 classes: business, entertainment, politics, sport, tech)
- **Preprocessing**: lowercasing, tokenization, stopword removal, lemmatization
- **Features**: TF-IDF (`max_features=5000`, unigrams+bigrams)
- **Model**: Multinomial Naive Bayes (`MultinomialNB`)
- **Artifacts**: `results/topic_classifier.pkl`, `results/learning_curves.png`
- **Performance**: ~97.5% test accuracy on held-out split; learning curves show no overfitting

### 3. Sentiment Analysis
- **Model**: NLTK VADER (`vader_lexicon`, pre-trained)
- **Output**: `[compound, pos, neg, neu]` scores per article
- **Rationale**: no labeled sentiment dataset required and fast inference; limitation is reduced nuance on formal/news text compared to domain-specific models

### 4. Scandal Detection
- **Keywords**: `pollution`, `deforestation`, `contamination`, `spill`, `emission`, `toxic waste` (unambiguous environmental terms; excludes natural-disaster-only terms like fire/flood/drought)
- **Embeddings**: SpaCy token vectors (`en_core_web_sm`) for sentence and keyword representations
- **Distance**: Euclidean distance between sentence and keyword vectors
- **Aggregation**: for each sentence containing an ORG entity, take mean of the 2 closest keyword distances; per article use the minimum of these sentence scores (`999.0` if no qualifying sentence)
- **Top 10**: exactly 10 articles with smallest `Scandal_distance` are flagged `Top_10=True` (deterministic tie-break by original index)

## How to Run

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords'); nltk.download('wordnet')"
```

1. Scrape articles:
   ```bash
   python3 scraper_news.py
   ```
2. Run NLP enrichment:
   ```bash
   python3 nlp_enriched_news.py
   ```
3. (Optional) Retrain topic model / regenerate learning curves:
   ```bash
   python3 results/training_model.py
   ```

## Notes / Limitations

- RSS feeds and article pages can change structure/availability; scraper includes fallbacks and URL de-duplication.
- VADER is a general pre-trained sentiment model and may miss domain-specific nuance in news.
- Scandal detection is keyword/embedding-distance based (unsupervised heuristic), so it can produce false positives/negatives depending on wording and entity context.