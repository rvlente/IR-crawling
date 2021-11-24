import itertools
import re
import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils.random import sample_without_replacement
import numpy as np
from nltk import ngrams
import time
import xgboost as xgb


exlude_words = ['www', 'index', 'html', 'htm', 'http', 'https']


def _extract_cctld(url):
    return [word for word in re.findall(r'[a-zA-Z]+', url) if word in ['nl', 'be', 'sr']]


def _extract_word_features(url):
    return [word for word in re.findall(r'[a-zA-Z]+', url) if len(word) > 2 and word not in exlude_words]


def _extract_trigram_features(url):
    return [ngram for word in re.findall(r'[a-zA-Z]+', url) for ngram in ngrams(word, 2) if len(word) > 2 and word not in exlude_words]


def load_dataset(take=None, split=0.75, feature_extractor=_extract_word_features):
    with open('/home/nils/Documents/school/information_retrieval/IR-crawling/cache/url_data.csv') as csvfile:
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
    # Load and split dataset
    X_train, X_test, y_train, y_test = load_dataset(
        take=10_000, split=0.1, feature_extractor=_extract_trigram_features)

    cu_time = time.time()
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set up parameters for xgboost
    param = {'max_depth': 6, 'eta': 0.1, 'objective': 'binary:logistic'}
    evallist = [(dtest, 'eval'), (dtrain, 'train')]

    bst = xgb.train(param, dtrain, num_boost_round=100)

    y_pred = bst.predict(dtest)
    y_pred = np.round(y_pred).astype(int)

    
    # clf = RandomForestClassifier(n_estimators=500)
    # clf = clf.fit(X_train, y_train)
    # Optional to get values faster but without probabilities?
    # print(np.sort(np.array(clf.coef_)))

    # y_pred = clf.predict(X_test)
    print("Time: ", time.time()-cu_time)
    print("hello")

    # Print metrics
    print('Precision:', precision_score(y_test, y_pred, pos_label=True))
    print('Recall:', recall_score(y_test, y_pred, pos_label=True))
    print('F-score:', f1_score(y_test, y_pred, pos_label=True))

    # Linear:
    # Precision: 0.8120264388199259
    # Recall: 0.8559048428207306
    # F - score: 0.8333884844473859

    # Time with N train data:
    #   10k:  0.31818270683288574
    #   1k:   0.00898289680480957
    #   0.1k: 0.000997304916381836

    # Kernel-Linear:
    # Precision: 0.8226427734047899
    # Recall: 0.8291694800810263
    # F - score: 0.8258932324506095

    # Time with N train data:
    #   10k:  54.11838150024414
    #   1k:   1.0571753978729248
    #   0.1k: 0.0049860477447509766

def load_dataset_only(take=None):
    with open('/home/nils/Documents/school/information_retrieval/IR-crawling/cache/url_data.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # Skip header.

        urls = []
        labels = []

        for url, is_dutch in itertools.islice(reader, take):
            urls.append(url)
            labels.append(is_dutch == 'True')
    return urls, labels

def getBestFeaturesNaive():
    urls, labels = load_dataset_only(10_000)

    vectorizer = CountVectorizer(ngram_range=(3,3), analyzer='char').fit(urls)
    # vectorizer = CountVectorizer(analyzer=_extract_word_features).fit(urls)
    featurized_data = vectorizer.transform(urls)
    vocabulary = vectorizer.vocabulary_

    counter = np.zeros((len(vectorizer.vocabulary_), 2))

    for feature_vector, label in zip(featurized_data, labels):
        f = feature_vector.tocoo()
        counter[f.col,int(label)] += 1

    sort_indices = np.argsort(list(vocabulary.values()))
    keys = np.array(list(vocabulary.keys()))[sort_indices]
    values = counter.tolist()

    feature_list = list(zip(keys, values))
    feature_list = sorted(feature_list, key=lambda x: (x[1][1] + 1)/(x[1][0] + 1), reverse=True)

    dutchest_features = np.array(feature_list[:100], dtype=object)
    undutchest_features = np.array(sorted(feature_list[len(feature_list)-100:], key=lambda x: (x[1][1] + 1)/(x[1][0] + 1)), dtype=object)

    counts = dutchest_features.tolist() + undutchest_features.tolist()

    keys = dutchest_features[:, 0].tolist() + undutchest_features[:, 0].tolist()
    values = list(range(len(keys)))
    dictionary = dict(zip(keys, values))

    return counts, dictionary

def testBestFeatures():
    cu_time = time.time()
    features, dictionary = getBestFeaturesNaive()
    print(features)
    print(dictionary)
    print("Time: ", time.time() - cu_time)


if __name__ == '__main__':
    main()
