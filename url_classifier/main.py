import itertools
import re
import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from nltk import ngrams
from ray import tune


exlude_words = ['www', 'index', 'html', 'htm', 'http', 'https']


def _extract_cctld(url):
    return [word for word in re.findall(r'[a-zA-Z]+', url) if word in ['nl', 'be', 'sr']]


def _extract_word_features(url):
    return [word for word in re.findall(r'[a-zA-Z]+', url) if len(word) > 2 and word not in exlude_words]


def _extract_trigram_features(url):
    return [ngram for word in re.findall(r'[a-zA-Z]+', url) for ngram in ngrams(word, 2) if len(word) > 2 and word not in exlude_words]


def load_dataset(take=None, split=0.75, feature_extractor=_extract_trigram_features):
    with open('/home/nils/Documents/school/information_retrieval/IR-crawling/cache/url_data.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # Skip header.

        urls = []
        labels = []

        for url, is_dutch in itertools.islice(reader, take):
            urls.append(url)
            labels.append(is_dutch == 'True')

    urls_train, urls_test, labels_train, labels_test = train_test_split(urls, labels, test_size=split)

    vectorizer = CountVectorizer(analyzer=feature_extractor)
    vectorizer.fit(urls_train)

    X_train = vectorizer.transform(urls_train)
    X_test = vectorizer.transform(urls_test)

    return X_train, X_test, labels_train, labels_test

def train_function(config):
    X_train, X_test, y_train, y_test = load_dataset(
        take=10000, split=0.1, feature_extractor=_extract_trigram_features)

    clf = RandomForestClassifier(n_estimators=config['n_estimators'])
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    return {
        'precision': precision_score(y_test, y_pred, pos_label=True),
        'recall': recall_score(y_test, y_pred, pos_label=True),
        'f1': f1_score(y_test, y_pred, pos_label=True)
    }

def main():
    X_train, X_test, y_train, y_test = load_dataset(take=None, split=0.25, feature_extractor=_extract_trigram_features)

    # # clf = SVC(kernel='linear')
    clf = RandomForestClassifier(n_estimators=50)
    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print('Precision:', precision_score(y_test, y_pred, pos_label=True))
    print('Recall:', recall_score(y_test, y_pred, pos_label=True))
    print('F-score:', f1_score(y_test, y_pred, pos_label=True))

    # analysis = tune.run(
    #     train_function,
    #     config={
    #         'n_estimators': tune.grid_search([10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 150, 200])
    #     },
    # )

    # print(analysis.get_best_config(metric='recall', mode='max'))


if __name__ == '__main__':
    main()
