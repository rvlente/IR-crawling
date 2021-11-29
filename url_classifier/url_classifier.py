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
from sklearn.svm import SVC
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


memory = joblib.Memory(location="./cache/joblib_mem", verbose=0)

class UrlClassifier:

    def __init__(self, ngram_size=2, top_k_ngrams=200, n_estimators=500, classifier_type: str = "gradient_boosting") -> None:
        """
        :param ngram_size: The size of the ngrams to use
        :param top_k_ngrams: The number of ngrams to use
        :param n_estimators: The number of estimators for the classifier
        :param classifier_type: The type of classifier to use. options: "gradient_boosting", "SVM"
        """
        self._n = ngram_size
        self._k= top_k_ngrams

        self._top_k_grams: Optional[list[tuple]] = None

        if classifier_type == "gradient_boosting":
            self._classif: XGBClassifier = XGBClassifier(n_estimators=n_estimators, use_label_encoder=False, tree_method='gpu_hist', verbosity=0)
        elif classifier_type == "SVM":
            self._classif: SVC = SVC(gamma='auto', kernel='linear', probability=True)
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
        return self._classif.predict_proba(self._extract_features(urls))
    
    def predict_dutchiness(self, urls: list[str]) -> np.ndarray:
        return self.predict_proba(urls)[:,1]
    
    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'UrlClassifier':
        return joblib.load(path)

