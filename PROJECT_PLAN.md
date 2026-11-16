# NLP‑Enriched News Intelligence Platform – Project Plan & Explainer

This document explains **what the project does**, **how it was built**, and **how each part works** in both plain language and technical terms.

## 1. Big Picture

**Simple:** This project automatically collects recent news articles, figures out which companies are mentioned, what each article is about, whether the tone is positive/negative/neutral, and which articles might be about an environmental scandal.

**Technical:** An end-to-end NLP pipeline: RSS/web scraping → structured storage (JSONL) → enrichment (NER, supervised topic classification, lexicon-based sentiment analysis, embedding-distance heuristic for scandal ranking) → CSV output.

**End-to-end flow:**
1. `scraper_news.py` collects articles into `data/articles.jsonl`.
2. `nlp_enriched_news.py` reads them and adds NLP-derived fields.
3. Results/models are saved in `results/` (`enhanced_news.csv`, `topic_classifier.pkl`, `learning_curves.png`).

---

## 2. Project Structure & Data

| Path | Purpose |
|---|---|
| `scraper_news.py` | Fetches and stores news articles |
| `nlp_enriched_news.py` | Runs NLP enrichment pipeline |
| `results/training_model.py` | Trains topic classifier and plots learning curves |
| `data/articles.jsonl` | Scraped input data (one JSON object per line) |
| `results/enhanced_news.csv` | Final enriched dataset |
| `results/topic_classifier.pkl` | Saved TF-IDF vectorizer + classifier |
| `results/learning_curves.png` | Training vs validation accuracy plot |

**Scraped record schema (`data/articles.jsonl`):**
- `unique_id` (str/uuid), `url` (str), `date` (`YYYY-MM-DD`), `headline` (str), `body` (str)

**Enriched output schema (`results/enhanced_news.csv`):**
- `Unique ID`, `URL`, `Date scraped`, `Headline`, `Body`, `Org`, `Topics`, `Sentiment`, `Scandal_distance`, `Top_10`

---

## 3. Phase 1 – Scraper & Data Collection

**Simple:** The scraper visits news feeds, grabs recent articles (last 7 days), opens each article page to get the main text, avoids duplicates, and saves at least 300 articles.

**Technical:**
- **Sources:** Multiple RSS feeds (BBC News and The Guardian sections) to reach volume/diversity.
- **Date filter:** Parses RSS `pubDate`; keeps items within last 7 days.
- **De-duplication:** Tracks URLs already stored in `data/articles.jsonl` and in current run.
- **Body extraction:** Requests article URL and extracts text blocks/paragraphs via BeautifulSoup; falls back to RSS description/meta description if parsing fails.
- **Output format:** Appends JSONL records with `unique_id`, `url`, `date`, `headline`, `body`.
- **Progress output:** Prints `scraping <URL> - requesting/parsing/saved in <path>` per article.
- **Current state:** 300 unique recent articles collected.

**Why this matters:** The NLP pipeline quality depends on having real, recent, non-duplicate articles with usable body text (not just titles/snippets).

---

## 4. Phase 2 – Entity Detection (Companies/Organizations)

**Simple:** For each article, the system finds names of organizations/companies mentioned (like “BBC”, “Google”, etc.).

**Technical:**
- **Tool/model:** SpaCy `en_core_web_sm` named entity recognition (NER).
- **Input:** `headline + body`.
- **Target label:** `ORG` entities.
- **Post-processing:**
  - Blacklist obvious non-company/product false positives (e.g. iPhone, Galaxy, Pixel, Android, iPad, AirPods, MacBook, Chrome).
  - Overlap deduplication: when entity spans overlap/subsume, keep the longest span.
- **Output:** `Org` column as a list of detected organization strings (stored as string representation in CSV).
- **Console:** Prints detected count and sample org names per article.

**Limitations:** Small SpaCy model can miss entities or mislabel them; blacklist/dedup heuristics reduce but don’t eliminate errors.

---

## 5. Phase 3 – Topic Classification

**Simple:** The system learns from a labeled news dataset what “business/sport/tech/etc.” articles look like, then predicts the topic of each scraped article.

**Technical:**
- **Training data:** BBC News dataset (5 classes: `business`, `entertainment`, `politics`, `sport`, `tech`).
- **Training script:** `results/training_model.py` (downloads/extracts data, trains, saves artifacts).
- **Preprocessing:** lowercase, regex tokenization, stopword removal (NLTK), lemmatization (WordNet).
- **Features:** `TfidfVectorizer(max_features=5000, ngram_range=(1,2))`.
- **Classifier:** `MultinomialNB` (multinomial Naive Bayes), suited for sparse text/count-like features.
- **Evaluation:** train/test split with ~97.5% test accuracy; learning curves plotted to check generalization.
- **Artifacts:** `results/topic_classifier.pkl` (vectorizer + classifier), `results/learning_curves.png`.
- **Inference:** `nlp_enriched_news.py` applies same preprocessing/vectorization and predicts one topic per article.
- **Output:** `Topics` column stored as a list (e.g. `['business']`).

**Learning curves (jargon/simple):**
- **Simple:** Shows if the model is actually learning or just memorizing.
- **Technical:** Plots training and validation accuracy vs training size; small gap and both high/rising indicate low overfitting.

---

## 6. Phase 4 – Sentiment Analysis

**Simple:** Estimates whether an article sounds positive, negative, or neutral overall.

**Technical:**
- **Model:** NLTK VADER (`SentimentIntensityAnalyzer`, pre-trained lexicon).
- **Input:** `headline + body`.
- **Output:** `[compound, pos, neg, neu]`
  - `compound`: overall normalized score in [-1, 1] (negative to positive)
  - `pos/neg/neu`: proportions in [0, 1]
- **Stored as:** `Sentiment` list in CSV.

**Why pre-trained:** Avoids costly manual labeling and is quick to run; tradeoff is that VADER is tuned for social-media-like text and may miss nuance in formal news.

---

## 7. Phase 5 – Scandal Detection (Environmental)

**Simple:** Looks for articles where companies are mentioned near language similar to environmental harm (pollution, spills, toxic waste, etc.), then flags the 10 most suspicious ones.

**Technical:**
- **Keywords (unambiguous environmental terms):** `pollution`, `deforestation`, `contamination`, `spill`, `emission`, `toxic waste`.
- **Excluded examples:** ambiguous natural-disaster terms like `fire`, `flood`, `drought` to reduce false positives.
- **Representations:** SpaCy token vectors (`en_core_web_sm`, 300-dim vectors when available) for sentences and keywords.
- **Distance metric:** Euclidean distance between sentence embedding and keyword embedding.
  - **Simple:** smaller distance = more similar meaning/context.
- **Sentence selection:** Only sentences containing at least one detected `ORG` entity are considered.
- **Aggregation:**
  - Per sentence: mean of the `k=2` closest keyword distances.
  - Per article: minimum sentence score (most suspicious sentence dominates).
  - If no qualifying sentence: default large distance (`999.0`) so it ranks low.
- **Top 10 flagging:** Sort articles by `(Scandal_distance, original_index)` ascending and mark first 10 as `Top_10=True` (exactly 10, deterministic tie-break).
- **Output:** `Scandal_distance` (float), `Top_10` (bool).

**Limitations:** Heuristic/unsupervised method; depends on keyword choice, vector quality, and context. Can flag non-scandal environmental coverage or miss indirect scandals.

---

## 8. How to Run

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords'); nltk.download('wordnet')"
```

1. Scrape articles (stores `data/articles.jsonl`):  
   `python3 scraper_news.py`
2. Enrich articles (writes `results/enhanced_news.csv`):  
   `python3 nlp_enriched_news.py`
3. Optional: retrain topic model / regenerate curves:  
   `python3 results/training_model.py`

---

## 9. Current Results Snapshot

- `data/articles.jsonl`: 300 scraped articles (recent, unique URLs, bodies populated)
- `results/enhanced_news.csv`: 300 enriched rows, required columns present, **exactly 10** `Top_10=True`
- `results/topic_classifier.pkl` and `results/learning_curves.png`: trained topic model and learning curve evidence

This plan now serves as both implementation history and a read-later explainer in technical and simplified form.

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