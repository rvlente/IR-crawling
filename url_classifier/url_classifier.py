import argparse
from collections import defaultdict
from dataclasses import dataclass
from re import S
from typing import Iterable, Optional
from nltk.util import ngrams
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import nltk
import mlflow
from tqdm import tqdm
import joblib
import numpy as np
from xgboost.sklearn import XGBClassifier
import plotly.express as px
import functools
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
import time
import multiprocessing
from feature_extractors import *


memory = joblib.Memory(location="./cache/joblib_mem", verbose=0)

class UrlClassifier:

    def __init__(self, ngram_size=None, top_k_ngrams=200, n_estimators=500, classifier_type: str = "gradient_boosting", feature_type: str = "top_k_ngrams_size_n", use_gpu=True) -> None:
        """
        :param ngram_size: The size of the ngrams to use
        :param top_k_ngrams: The number of ngrams to use
        :param n_estimators: The number of estimators for the classifier
        :param classifier_type: The type of classifier to use. options: "gradient_boosting", "SVM", "LinearSVC", "ProbaLinearSVC"
        :param feature_type: The type of features to use. options: "top_k_ngrams_size_n", "ngrams_size_many"
        """
        if ngram_size is not None:
            self._n = ngram_size
        else:
            self._n = [2]
        self._k= top_k_ngrams

        self._top_k_grams: Optional[list[tuple]] = None

        if classifier_type == "gradient_boosting":
            tree_method = "gpu_hist" if use_gpu else "hist"
            self._classif: XGBClassifier = XGBClassifier(n_estimators=n_estimators, use_label_encoder=False, tree_method=tree_method, verbosity=0)
        elif classifier_type == "SVM":
            self._classif: SVC = SVC(gamma='auto', kernel='linear', probability=True)
        elif classifier_type == "ProbaLinearSVC":
            self._classif: CalibratedClassifierCV = CalibratedClassifierCV(LinearSVC())

        if feature_type == "top_k_ngrams_size_n":
            self._n = self._n[0]
            self._ft_extractor: Extract_top_k_grams_size_n = Extract_top_k_grams_size_n()
        elif feature_type == "ngrams_size_many":
            self._ft_extractor: Extract_n_grams_size_all = Extract_n_grams_size_all(self._n, self._k)

        self._label_encoder = LabelEncoder()

    def _extract_top_k_grams(self, train_urls: list[str]):

        all_ngrams = (nltk.ngrams(url, self._n) for url in train_urls)
        all_ngrams = [item for sub_list in all_ngrams for item in sub_list]

        ngrams_freq = nltk.FreqDist(all_ngrams)

        self._top_k_grams = [x[0] for x in ngrams_freq.most_common(self._k)]


    def _extract_features(self, urls: Iterable[str], use_tqdm=True, parallel_feature_extraction=True) -> list[list[int]]:

        if self._top_k_grams is None:
            raise ValueError("Top k ngrams not set, please call fit first")

        def extract_fn(url: str, top_k_grams, n):
            ngrams_in_url = nltk.FreqDist(nltk.ngrams(url, n))
            return [ngrams_in_url.get(g, 0) for g in top_k_grams]
            
        if parallel_feature_extraction:
            try:
                n_cpus = multiprocessing.cpu_count()
                result = Parallel(n_jobs=n_cpus, batch_size=len(urls)//n_cpus)(delayed(extract_fn)(url, self._top_k_grams, self._n) for url in urls)
            except ValueError:
                result = [extract_fn(url, self._top_k_grams, self._n) for url in urls]
        else:
            result = [extract_fn(url, self._top_k_grams, self._n) for url in urls]


        # for url in tqdm(urls, desc="Extracting features", disable=not use_tqdm):
        #     ngrams_in_url = nltk.FreqDist(nltk.ngrams(url, self._n))
        #     result.append(
        #         [ngrams_in_url.get(g, 0) for g in self._top_k_grams]
        #     )
            
        return result

    def fit(self, train_urls: list[str], train_labels: list, parallel_feature_extraction=True) -> 'UrlClassifier':
        self._extract_top_k_grams(train_urls)
        train_features = self._extract_features(train_urls, parallel_feature_extraction=parallel_feature_extraction)
        # train_features = np.array(train_features)
        


        self._label_encoder.fit(train_labels)
        train_labels = self._label_encoder.transform(train_labels)

        self._classif.fit(train_features, train_labels)

        return self

    def predict(self, urls: list[str]) -> np.ndarray:
        return self._label_encoder.inverse_transform(self._classif.predict(self._extract_features(urls)))

    def predict_proba(self, urls: list[str]) -> np.ndarray:
        return self._classif.predict_proba(self._extract_features(urls, parallel_feature_extraction=False))
    
    def predict_dutchiness(self, urls: list[str]) -> np.ndarray:
        return self.predict_proba(urls)[:,1]
    
    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'UrlClassifier':
        return joblib.load(path)

    def test(self, dataPath='url_data_with_context_full.csv'):
        # Load data
        with open(dataPath) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader)  # Skip header.

            urls = []
            labels = []

            for url, is_dutch, relative_url, text, parent_text in itertools.islice(reader, take):
                urls.append(url)
                labels.append(is_dutch == 'True')

        # Embed data
        vectorizer = CountVectorizer(analyzer=feature_extractor)
        features = vectorizer.fit_transform(urls)

        # Split in train and test sets
        X_train, X_test, y_train, y_test = train_test_split(features, labels, shuffle=True, train_size=split)

        # Test runtime
        cu_time = time.time()
        y_pred = self.predict(X_test)
        print("Time: ", time.time() - cu_time)

        # Print metrics
        print('Precision:', precision_score(y_test, y_pred, pos_label=True))
        print('Recall:', recall_score(y_test, y_pred, pos_label=True))
        print('F-score:', f1_score(y_test, y_pred, pos_label=True))

