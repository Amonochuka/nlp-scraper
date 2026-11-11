# NLP‑Enriched News Intelligence Platform – Project Plan

## Overview
Build a platform that reads scraped news articles, identifies companies (ORG), classifies topics, gauges sentiment, and flags potential environmental scandals. The project is split into seven phases; each includes what is already done and the next concrete steps.

---

## Phase 1 – Scraper & data
**What’s complete**
- `scraper_news.py` fetches an RSS feed and stores records in `data/articles.jsonl` with schema: `unique_id`, `url`, `date`, `headline`, `body`.
- Project skeleton (`.gitignore`, empty `data/` and `results/` with `.gitkeep`).
- Initial commit with docs (`README.md`, `agents.md`).

**What’s next**
- Run the scraper in your own network until `data/articles.jsonl` contains **≥300 articles** from the last week.
- (Optional) add a `--feed` CLI flag to swap sources.
- Verify the console output matches the deliverable format:
  ```
  scraping <URL> - requesting ...
  scraping <URL> - parsing ...
  scraping <URL> - saved in <path>
  ```

**Learning‑prompt tie‑in**
> “Explain how an RSS feed works, then ask me what could make a feed‑scraper fail or double‑count articles.”

---

## Phase 2 – Entity detection (SpaCy NER)
**What’s complete**
- `nlp_enriched_news.py` runs SpaCy NER, extracts `ORG` entities, prints the required line:
  `Detected <X> companies which are …`
- Tested with dummy articles; CSV output schema confirmed.

**What’s next**
- Refine `extract_orgs` to avoid false positives (product names like `iPhone`, `Galaxy`, `Pixel`).
- Add a simple deduplication step (keep only the longest ORG span when sub‑spans overlap).
- Small blacklist of product names; normalise whitespace & `&` symbols.

**Learning‑prompt tie‑in**
> “Explain how named entity recognition finds organizations in text, then ask me what could make SpaCy miss or mislabel a company name.”

---

## Phase 3 – Topic classification (BBC dataset + learning curves)
**What’s complete**
- Placeholder `classify_topic` uses keyword mapping.

**What’s next**
- Download the **BBC News dataset** (`http://mlg.ucd.ie/datasets/bbc.html`).
- Split into train/test (80/20 or 90/10).
- Preprocess: lowercase, tokenize, remove stopwords, lemmatize.
- Vectorize with `TfidfVectorizer`.
- Train a `MultinomialNB` or `LogisticRegression` classifier.
- Save model as `results/topic_classifier.pkl`.
- Plot **learning curves** (`learning_curves.png`): training vs. validation accuracy as training‑set size grows. The plot must demonstrate **no overfitting** (small gap, both curves rising).
- Report test accuracy (target **>95%** on BBC).

**Learning‑prompt tie‑in**
> “Walk me through building a topic classifier on a labelled news dataset, then quiz me on what a learning curve shows about overfitting.”

---

## Phase 4 – Sentiment analysis (pre‑trained VADER)
**What’s complete**
- Placeholder `analyze_sentiment` returns `[0.0, 0.0, 0.0]`.

**What’s next**
- Install `nltk` and download the VADER lexicon:
  `python3 -c "import nltk; nltk.download('vader_lexicon')"`.
- Use `nltk.SentimentIntensityAnalyzer` to compute `compound`, `pos`, `neg`, `neu` scores.
- Output format: `Sentiment` column as a list/float per the CSV schema, and the console line
  `The article <title> has a <sentiment> sentiment`.
- Discuss in the README why a pre‑trained model was chosen (labelling cost, speed, limitations on formal news).

**Learning‑prompt tie‑in**
> “Explain why we use a pre‑trained sentiment model instead of training our own, then ask me about the limits of applying a general sentiment model to news.”

---

## Phase 5 – Scandal detection (word embeddings + distance)
**What’s complete**
- Placeholder `flag_scandal` returns `Scandal_distance=0.0`, `Top_10=False`.

**What’s next**
- Define an **unambiguous keyword set**: `pollution`, `deforestation`, `contamination`, `spill`, `emission`, `toxic waste`. Exclude natural‑disaster words (`fire`, `flood`, `drought`).
- Compute sentence embeddings (SpaCy `embedding` or a lightweight Sentence‑BERT model).
- For each sentence containing an `ORG` entity, compute the distance (cosine similarity or Euclidean) to each keyword vector.
- **Aggregate** per‑article: use *mean of the k‑closest* (k=2). Justify this choice in the README.
- Rank articles by the unified `Scandal_distance` (smallest = most suspicious) and flag the **top 10** as `Top_10=True`.
- Print the line `Environmental scandal detected for <entity>` when an entity is involved.

**Learning‑prompt tie‑in**
> “Explain how word embeddings let me measure how close a sentence is to the idea of an environmental scandal, then ask me how I would choose keywords without triggering false positives.”

---

## Phase 6 – README & documentation
**What’s complete**
- `README.md` and `agents.md` already committed.

**What’s next**
- Summarise the embeddings/model choices (why SpaCy `en_core_web_sm`, why VADER, why cosine distance, why k‑closest mean).
- List known limitations (e.g., VADER tuned on social media, ambiguous disaster keywords).
- Add a short **“How to run”** section.

---

## Phase 7 – Full end‑to‑end test
**What’s complete**
- Pipeline runs on dummy `data/articles.jsonl`; `enhanced_news.csv` has the required columns and row count.

**What’s next**
- After populating `data/articles.jsonl` with **≥300 articles** (Phase 1), run:
  `python3 nlp_enriched_news.py`
- Verify the CSV `enhanced_news.csv` has exactly the required columns and row‑count.
- Check the console output matches the deliverable format line‑for‑line.
- Commit the final `results/enhanced_news.csv` (or leave untracked if you prefer).

---

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