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
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
import sklearn
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
from .feature_extractors import Extract_top_k_grams_size_n, Extract_top_k_grams_size_many
import csv
import itertools

memory = joblib.Memory(location="./cache/joblib_mem", verbose=0)

class ContextClassifier:

    def __init__(self, ngram_size=None, top_k_ngrams=200, n_estimators=500, classifier_type: str = "gradient_boosting", feature_type: str = "top_k_ngrams_size_n", use_gpu=True) -> None:
        """
        :param context_size: The maximum size of the context to use
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
        self._k = top_k_ngrams

        self._top_k_grams: Optional[list[tuple]] = None
        self.classifier_type = classifier_type
        self.feature_type = feature_type

        if classifier_type == "gradient_boosting":
            tree_method = "gpu_hist" if use_gpu else "hist"
            self._classif: XGBClassifier = XGBClassifier(n_estimators=n_estimators, use_label_encoder=False, tree_method=tree_method, verbosity=0)
        elif classifier_type == "SVM":
            self._classif: SVC = SVC(gamma='auto', kernel='linear', probability=True)
        elif classifier_type == "ProbaLinearSVC":
            self._classif: CalibratedClassifierCV = CalibratedClassifierCV(LinearSVC(max_iter=10000))

        if feature_type == "top_k_ngrams_size_n":
            self._n = self._n[0]
            self._ft_extractor: Extract_top_k_grams_size_n = Extract_top_k_grams_size_n(self._n, self._k)
        elif feature_type == "top_k_ngrams_size_many":
            self._ft_extractor: Extract_top_k_grams_size_many = Extract_top_k_grams_size_many(self._n, self._k)

        self._label_encoder = LabelEncoder()

    def fit(self, train_contexts: list[str]=None, train_labels: list=None, parallel_feature_extraction=True, dataPath=None) -> 'ContextClassifier':
        if train_contexts is None and train_labels is None and dataPath is None:
            raise ValueError("No data is given. Specify either train_contexts and train_labels or provide a path name")
        if train_contexts is not None and train_labels is not None and dataPath is not None:
            raise ValueError("Both (train_contexts, train_labels) and path name are specified. Please specify only one to avoid ambiguity")

        if dataPath is not None:
            train_contexts, train_labels = self.loadData(dataPath)
        self._ft_extractor.prepare(train_contexts)
        train_features = self._ft_extractor.extract_features(train_contexts, parallel_feature_extraction=parallel_feature_extraction)

        self._label_encoder.fit(train_labels)
        train_labels = self._label_encoder.transform(train_labels)

        self._classif.fit(train_features, train_labels)

        return self

    def predict(self, contexts: list[str]) -> np.ndarray:
        # return self._label_encoder.inverse_transform(self._classif.predict(self._extract_features(contexts)))
        return self._label_encoder.inverse_transform(self._classif.predict(self._ft_extractor.extract_features(contexts)))

    def predict_proba(self, contexts: list[str]) -> np.ndarray:
        # return self._classif.predict_proba(self._extract_features(contexts, parallel_feature_extraction=False))
        return self._classif.predict_proba(self._ft_extractor.extract_features(contexts, parallel_feature_extraction=False))

    def predict_dutchiness(self, contexts: list[str]) -> np.ndarray:
        return self.predict_proba(contexts)[:,1]
    
    def save(self, path: str) -> None:
        joblib.dump(self, path + " " + self.classifier_type + " " + self.feature_type)

    @classmethod
    def load(cls, path: str) -> 'ContextClassifier':
        return joblib.load(path)

    def test(self, dataPath, split=0.9, take=None):
        # Load data
        contexts, labels = self.loadData(dataPath, take)

        # Split in train and test sets
        X_train, X_test, y_train, y_test = train_test_split(contexts, labels, shuffle=True, train_size=split)

        # Test for training time
        cu_time = time.time()
        self.fit(X_train, y_train)
        seconds = time.time() - cu_time
        seconds_per_hundred_thousand_training_samples = seconds * (100000/len(X_train))

        # Test prediction time
        cu_time = time.time()
        y_pred = self.predict(X_test)
        seconds = time.time() - cu_time
        seconds_per_hundred_thousand_predictions = seconds * (100000/len(X_test))

        # Print metrics
        precision = precision_score(y_test, y_pred, pos_label=True)
        recall = recall_score(y_test, y_pred, pos_label=True)
        fscore = f1_score(y_test, y_pred, pos_label=True)

        precision = precision.item()
        recall = recall.item()
        fscore = fscore.item()

        return {"classifier_type": self.classifier_type,
                "feature_type": self.feature_type,
                "seconds_training": seconds_per_hundred_thousand_training_samples,
                "seconds_prediction": seconds_per_hundred_thousand_predictions,
                "precision": precision, "recall": recall, "fscore": fscore}

    def loadData(self, dataPath, take=None):
        df = pd.read_parquet(dataPath, engine='pyarrow')

        if take is None:
            contexts = df["url_context"]
            labels = df["is_dutch"]
        else:
            contexts = df["url_context"][:min(take, len(df["url_context"])-1)]
            labels = df["is_dutch"][:min(take, len(df["url_context"])-1)]

        return contexts, labels
