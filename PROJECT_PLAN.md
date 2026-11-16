# NLP‑Enriched News Intelligence Platform – Project Plan

## Overview
Build a platform that scrapes news articles, identifies companies (ORG), classifies topics, gauges sentiment, and flags potential environmental scandals.

## Status: Completed

### Phase 1 – Scraper & data
- `scraper_news.py` fetches from multiple RSS feeds (BBC News and The Guardian sections), de-duplicates URLs, filters to the last 7 days, fetches article pages for body text, and stores records in `data/articles.jsonl` (`unique_id`, `url`, `date`, `headline`, `body`).
- Dataset populated with **300 unique articles** (real URLs, recent dates, non-empty bodies).

### Phase 2 – Entity detection (SpaCy NER)
- `nlp_enriched_news.py` extracts `ORG` entities from headline + body using `en_core_web_sm`.
- Post-processing includes blacklist filtering for common product-name false positives and overlap deduplication (longest span kept).

### Phase 3 – Topic classification (BBC dataset + learning curves)
- Trained TF-IDF + `MultinomialNB` classifier on the BBC News dataset; saved as `results/topic_classifier.pkl`.
- `results/training_model.py` and `results/learning_curves.png` document training/evaluation (high test accuracy, no overfitting indicated by learning curves).

### Phase 4 – Sentiment analysis (pre-trained VADER)
- Uses NLTK VADER (`vader_lexicon`) to output `[compound, pos, neg, neu]` per article.

### Phase 5 – Scandal detection (embeddings + distance)
- Environmental keyword set: `pollution`, `deforestation`, `contamination`, `spill`, `emission`, `toxic waste`.
- Computes Euclidean distances between sentence embeddings and keyword vectors (SpaCy vectors) for sentences containing ORG entities.
- Aggregates per sentence as mean of k=2 closest keyword distances, per article as minimum sentence score.
- Flags **exactly top 10** articles by smallest `Scandal_distance` (`Top_10=True`, deterministic tie-break).

### Phase 6 – Documentation
- `README.md` documents structure, model choices/embeddings/distance rationale, limitations, and how to run.
- Temporary instruction/scaffold doc removed after completion.

### Phase 7 – End-to-end run
- `python3 nlp_enriched_news.py` executed on `data/articles.jsonl` (300 articles) and generated `results/enhanced_news.csv` with required columns and correct `Top_10` count.

## Quick‑start checklist (run in order)
1. **Run the scraper** in your own network (BBC/CNN/RSS‑compatible feed) until `data/articles.jsonl` has 300+ articles from the last week.
2. **Improve entity‑extraction** – fine‑tune `extract_orgs` to avoid product‑name false positives.
3. **Implement topic classification** – BBC dataset, train & export `topic_classifier.pkl`, plot learning curves.
4. **Add VADER sentiment** – one‑liner after `nltk` download.
5. **Build scandal detection** – keyword list, embeddings, k‑closest mean aggregation, top‑10 flag.
6. **Write the README** – explain all model/embedding choices and why.
7. **Final test** – run the whole pipeline, verify CSV & console output, commit.

---
*This file is intended as a permanent reference in case the session is lost. Keep it alongside `README.md` and `agents.md` in the repo root.*