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