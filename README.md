# NLP-Enriched News Intelligence Platform

Project to build an NLP-enriched News Intelligence platform that reads scraped news articles, identifies companies mentioned, classifies each article's topic, gauges its sentiment, and flags articles that may describe an environmental scandal involving those companies.

## Project Structure

- `scraper_news.py` - Fetches and stores news articles from RSS feeds
- `nlp_enriched_news.py` - Main NLP pipeline for entity detection, topic classification, sentiment analysis, and scandal detection
- `data/` - Scraped news articles stored in JSON format
- `results/` - NLP pipeline outputs including trained models, CSV outputs, and learning curves

## Deliverables

- At least 300 news articles scraped and stored
- NLP enrichment pipeline processing all articles
- `enhanced_news.csv` with entity, topic, sentiment and scandal detection results
- Trained topic classifier with learning curves proving no overfitting

## Model Choices Summary

### Topic Classifier
- **Algorithm**: Multinomial Naive Bayes (MultinomialNB)
- **Dataset**: BBC News dataset (2225 articles, 5 categories: business, entertainment, politics, sport, tech)
- **Features**: TF-IDF vectorization with max 5000 features, unigrams & bigrams
- **Preprocessing**: Lowercasing, NLTK stopword removal, WordNet lemmatization
- **Pre-trained model**: Saved as `results/topic_classifier.pkl` (includes fitted vectorizer + classifier)
- **Test accuracy**: 97.5% on held-out test set (exceeds >95% target)
- **Learning curves**: Plot demonstrates no overfitting (small gap between training and validation curves, both rising as training size increases)
- **Why this model**: Lightweight, fast inference, works well with high-dimensional text data (TF-IDF), no overfitting observed

### Sentiment Analysis
- **Algorithm**: VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Lexicon**: Pre-trained ~7,500 word sentiment lexicon from nltk
- **Scores**: [compound, pos, neg, neu] ranging from -1 to +1 (compound) and 0 to 1 (individual scores)
- **Pre-trained model**: Downloaded via `nltk.download('vader_lexicon')`, no training required
- **Why this model**: Zero labelling cost, fast inference, specifically designed for text with social media characteristics, works adequately on formal news; limitation: tuned on social media text, may not capture all formal news nuances

### Entity Detection (NER)
- **Model**: SpaCy `en_core_web_sm`
- **Entity type**: ORG (organizations)
- **Post-processing**: 
  - Blacklist of product-name false positives: iPhone, Galaxy, Pixel, Android, iPad, watch, AirPods, MacBook, pro, Chrome
  - Deduplication: keeps longest ORG span when sub-spans overlap
- **Why this model**: Small, fast, good enough ORG detection for news articles; product blacklist improves accuracy by filtering common consumer product names that SpaCy sometimes tags as organizations

### Scandal Detection
- **Method**: Word embedding distance computation
- **Keywords**: pollution, deforestation, contamination, spill, emission, toxic waste (6 unambiguous environmental keywords)
- **Excluded**: fire, flood, drought (natural disaster words to avoid false positives)
- **Distance metric**: Euclidean distance between SpaCy word vectors (300-dimensional)
- **Aggregation**: For each sentence with ORG entity, compute mean of k=2 closest keyword distances; aggregate across sentences: minimum mean distance
- **Top 10 flagging**: Articles ranked by Scandal_distance (smallest = most suspicious), exactly top 10 flagged as Top_10=True
- **Why this approach**: Unsupervised, no labelled training data needed; keyword-based approach is transparent and controllable; limitation: depends on keyword selection and SpaCy vector quality

## How to Run

1. **Run the scraper** to collect 300+ articles:
   ```bash
   python3 scraper_news.py
   ```

2. **Run the NLP pipeline**:
   ```bash
   python3 nlp_enriched_news.py
   ```

3. **Output**: Enriched data written to `results/enhanced_news.csv`
   - CSV columns: Unique ID, URL, Date scraped, Headline, Body, Org, Topics, Sentiment, Scandal_distance, Top_10

4. **Results located in**:
   - `results/topic_classifier.pkl` - Trained topic classification model
   - `results/learning_curves.png` - Learning curves showing no overfitting
   - `results/enhanced_news.csv` - Pipeline output with all enrichment results