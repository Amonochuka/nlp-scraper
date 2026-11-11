# NLP Scraper Project - Instruction Guide
> This file contains step-by-step instructions for building the NLP-enriched News Intelligence platform.
> Delete this file once the project is complete.

## Project Structure
```
/nlp-scraper/
├── data/                  # Scraped news data storage
├── results/               # NLP results and models
│   ├── training_model.py
│   ├── topic_classifier.pkl
│   ├── enhanced_news.csv
│   └── learning_curves.png
├── nlp_enriched_news.py   # Main NLP pipeline script
├── scraper_news.py        # Scraper script
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── agents.md              # THIS FILE WILL BE DELETED AFTER PROJECT COMPLETE
```

## Step 1: Scraper (scraper_news.py)
### Goal: Fetch and store at least 300 news articles

**1.1 Choose a news website that is easy to scrape**
- Select a news source with stable HTML structure
- Avoid sites with strict scraping policies or requiring authentication
- BBC, Reuters, or similar reputable sources are good candidates

**1.2 Scraping implementation**
- Use `requests` to fetch page content
- Use `BeautifulSoup` (or lxml) to parse HTML
- Extract from each article:
  - URL (unique)
  - Date scraped
  - Headline
  - Body text
- Clean body text (remove boilerplate, nav, ads)

**1.3 Data storage**
- Create `data/` directory structure
- Option A: One JSON/CSV file per day
- Option B: SQL database (sqlite recommended)
- Each record must contain: unique_id, url, date, headline, body
- Store at least 300 articles from the last week

**1.4 Scraper execution**
```bash
python scraper_news.py
# Expected output:
# scraping <URL> - requesting ...
# scraping <URL> - parsing ...
# scraping <URL> - saved in <path>
```

### Reminders (delete after project complete):
- [ ] Remove any AI/scaffold comments from code
- [ ] Delete this agents.md file
- [ ] Clean up temporary debug scripts

## Step 2: NLP Engine (nlp_enriched_news.py)
### Goal: Process scraped articles through NLP pipeline

**2.1 Entities Detection (SpaCy NER)**
- Load SpaCy `en_core_web_sm` or `en_core_web_md`
- Run NER on headline + body text
- Filter for `ORG` entity type (companies/organizations)
- Store list of detected orgs per article
- Output: `Detected X companies which are company1 and company2`

**2.2 Topic Detection (Classifier)**
- Download BBC News dataset from http://mlg.ucd.ie/datasets/bbc.html
- Split into train/test sets (80/20 or 90/10)
- Preprocess text: lowercase, tokenize, remove stopwords, lemmatize
- Feature extraction: `TfidfVectorizer` or `CountVectorizer`
- Train classifier: `MultinomialNB`, `SGDClassifier`, or `LogisticRegression`
- Save model as `results/topic_classifier.pkl`
- Generate learning curves plot: `results/learning_curves.png`
- Model should achieve >95% accuracy on test set
- No overfitting (gap between train/test accuracy should be small)

**2.3 Sentiment Analysis (NLTK pre-trained)**
- Use NLTK VADER or pre-trained sentiment model
- Classify each article as positive/negative/neutral
- Output sentiment as float (polarity) or list [pos, neg, neu]
- Note: Use pre-trained model (see README for rationale)

**2.4 Scandal Detection**
- Define environmental disaster keywords:
  - pollution, deforestation, contamination, spill, emission, poison
  - Avoid ambiguous words (fire, flood in natural context)
- Compute word embeddings (use SpaCy `embedding` or Sentence-BERT)
- For each sentence containing an ORG entity:
  - Compute embedding of sentence
  - Compute distance to keyword embeddings
  - Options: cosine similarity, Euclidean distance, Manhattan distance
- Explain in README.md: embeddings chosen and why
- Explain distance/similarity chosen and why
- Save unified metric: unify all per-sentence distances into one score per article
- Flag top 10 articles with lowest/highest distances
- Output: `Scandal_distance` per article, `Top_10` boolean flag

**2.5 Output structure (enhanced_news.csv)**
Each article row contains:
- Unique ID (uuid or int)
- URL (str)
- Date scraped (date)
- Headline (str)
- Body (str)
- Org (list str)
- Topics (list str)
- Sentiment (list float or float)
- Scandal_distance (float)
- Top_10 (bool)

### Execution
```bash
python nlp_enriched_news.py
# Should output enrichment progress per article
```

### Reminders (delete after project complete):
- [ ] Delete this agents.md file
- [ ] Remove any placeholder/debug code
- [ ] Ensure all output files are in results/

## Step 3: Requirements
### dependencies (requirements.txt)
```
requests>=2.28.0
beautifulsoup4>=4.11.0
spacy>=3.0.0
nltk>=3.7.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
```

### Optional (download commands)
Run once to download SpaCy model:
```bash
python -m spacy download en_core_web_sm
```

Run once to download NLTK data:
```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

## Step 4: Run the Full Pipeline
1. Run scraper: `python scraper_news.py` until 300+ articles stored
2. Run NLP pipeline: `python nlp_enriched_news.py`
3. Verify `results/enhanced_news.csv` has correct structure
4. Check `results/learning_curves.png` shows proper training
5. Review flagged top 10 scandal articles

## Step 5: Submission Checklist
- [ ] At least 300 articles scraped and stored in data/
- [ ] `scraper_news.py` works and logs scraping progress
- [ ] `nlp_enriched_news.py` processes all articles
- [ ] `results/enhanced_news.csv` has correct column structure
- [ ] `results/topic_classifier.pkl` trained and saved
- [ ] `results/learning_curves.png` generated
- [ ] Top 10 scandal-detected articles flagged
- [ ] README.md explains embeddings and distance metrics
- [ ] agents.md deleted