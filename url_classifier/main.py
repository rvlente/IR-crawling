import itertools
import re
import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from nltk import ngrams


exlude_words = ['www', 'index', 'html', 'htm', 'http', 'https']


def _extract_cctld(url):
    return [word for word in re.findall(r'[a-zA-Z]+', url) if word in ['nl', 'be', 'sr']]


def _extract_word_features(url):
    return [word for word in re.findall(r'[a-zA-Z]+', url) if len(word) > 2 and word not in exlude_words]


def _extract_trigram_features(url):
    return [ngram for word in re.findall(r'[a-zA-Z]+', url) for ngram in ngrams(word, 3) if len(word) > 2 and word not in exlude_words]


def load_dataset(take=None, split=0.75, feature_extractor=_extract_word_features):
    with open('url_data.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # Skip header.

        urls = []
        labels = []

        for url, is_dutch in itertools.islice(reader, take):
            urls.append(url)
            labels.append(is_dutch == 'True')

    vectorizer = CountVectorizer(analyzer=feature_extractor)
    features = vectorizer.fit_transform(urls)

    return train_test_split(features, labels, shuffle=True, train_size=split)


def main():
    X_train, X_test, y_train, y_test = load_dataset(
        take=10000, split=0.1, feature_extractor=_extract_trigram_features)

    clf = SVC(kernel='linear')
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print('Precision:', precision_score(y_test, y_pred, pos_label=True))
    print('Recall:', recall_score(y_test, y_pred, pos_label=True))
    print('F-score:', f1_score(y_test, y_pred, pos_label=True))


if __name__ == '__main__':
    main()
