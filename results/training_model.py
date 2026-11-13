"""
Training script for the topic classifier.

This script trains a MultinomialNB classifier on the BBC News dataset
and saves the model and learning curves for the pipeline.

The trained model is saved as topic_classifier.pkl
The learning curves are saved as learning_curves.png

Author: NLP Scraper Project
"""

import zipfile
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer


def download_data():
    """Download BBC News fulltext dataset."""
    import urllib.request

    url = 'http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip'
    print(f'Downloading BBC dataset from {url}...')
    resp = urllib.request.urlopen(url, timeout=120)
    data = resp.read()

    with open('/tmp/bbc-fulltext.zip', 'wb') as f:
        f.write(data)
    print(f'Downloaded {len(data)} bytes')


def extract_articles(zip_path='/tmp/bbc-fulltext.zip'):
    """Extract article texts and labels from the BBC dataset."""
    category_names = ['business', 'entertainment', 'politics', 'sport', 'tech']

    texts = []
    labels = []

    with zipfile.ZipFile(zip_path, 'r') as z:
        all_files = z.namelist()

        for fname in all_files:
            m = re.match(r'bbc\/([a-z]+)\/\d+\.txt', fname)
            if m:
                cat = m.group(1)
                if cat in category_names:
                    idx = category_names.index(cat)
                    text = z.read(fname).decode('utf-8', errors='replace')
                    texts.append(text)
                    labels.append(idx)

    print(f'Total articles extracted: {len(texts)}')
    print(f'Category distribution: ')
    from collections import Counter
    counts = Counter(labels)
    for name, idx in [('business', 0), ('entertainment', 1), ('politics', 2), ('sport', 3), ('tech', 4)]:
        print(f'  {name}: {counts.get(idx, 0)}')

    return texts, labels


def preprocess_text(text):
    """Preprocess text same as training: lowercase, tokenization, stopword removal, lemmatization."""
    import re

    text = text.lower()
    tokens = re.findall(r'\b[a-z]+\b', text)
    stop_words = set(nltk_stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 3]
    return ' '.join(tokens)


def train_model(texts, labels):
    """Train the MultinomialNB topic classifier."""
    # Split train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Preprocess
    X_train_clean = [preprocess_text(t) for t in X_train]
    X_test_clean = [preprocess_text(t) for t in X_test]

    # Vectorize with TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train_clean)
    X_test_vec = vectorizer.transform(X_test_clean)

    # Train MultinomialNB
    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f'Test accuracy: {acc:.4f} ({acc * 100:.1f}%)')

    # Learning curves
    train_sizes = np.arange(100, len(X_train) + 1, 100)
    train_accuracies = []
    val_accuracies = []

    for size in train_sizes:
        clf_size = MultinomialNB()
        clf_size.fit(X_train_vec[:size].toarray(), y_train[:size])
        train_acc = clf_size.score(X_train_vec[:size].toarray(), y_train[:size])
        val_acc = clf_size.score(X_test_vec, y_test)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_accuracies, 'o-', label='Training accuracy')
    plt.plot(train_sizes, val_accuracies, 'o-', label='Validation accuracy')
    plt.xlabel('Training set size')
    plt.ylabel('Accuracy')
    plt.title('Learning Curves - Topic Classifier (BBC News)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    return clf, vectorizer, acc, train_sizes, train_accuracies, val_accuracies


def save_artifacts(clf, vectorizer, train_sizes, train_accuracies, val_accuracies, acc):
    """Save the model and learning curves."""
    import joblib
    import os

    # Save model
    model_data = {'vectorizer': vectorizer, 'classifier': clf}
    joblib.dump(model_data, 'results/topic_classifier.pkl')
    print(f'Model saved to results/topic_classifier.pkl')

    # Save learning curves
    plt.savefig('results/learning_curves.png')
    print(f'Learning curves saved to results/learning_curves.png')

    # Print summary
    print(f'\n=== Training Summary ===')
    print(f'Test accuracy: {acc * 100:.1f}% (target: >95%)')
    print(f'No overfitting: {"Yes" if abs(train_accuracies[-1] - val_accuracies[-1]) < 0.1 else "No"}')


def main():
    """Main training pipeline."""
    # Download data
    download_data()

    # Extract articles
    texts, labels = extract_articles()

    # Train model
    clf, vectorizer, acc, train_sizes, train_accuracies, val_accuracies = train_model(texts, labels)

    # Save artifacts
    save_artifacts(clf, vectorizer, train_sizes, train_accuracies, val_accuracies, acc)

    print('\nTraining complete!')


if __name__ == '__main__':
    main()