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

memory = joblib.Memory(location="./cache/joblib_mem", verbose=0)

# TODO manual label encoding
class UrlClassifier:

    def __init__(self, ngram_size=2, top_k_ngrams=200, n_estimators=500) -> None:
        self._n = ngram_size
        self._k= top_k_ngrams

        self._top_k_grams: Optional[list[tuple]] = None
        self._classif: XGBClassifier = XGBClassifier(n_estimators=n_estimators, use_label_encoder=False, tree_method='gpu_hist', verbosity=0)
        self._label_encoder = LabelEncoder()

    def _extract_top_k_grams(self, train_urls: list[str]):
        all_ngrams = [ng for url in train_urls for ng in nltk.ngrams(url, self._n)]
        ngrams_freq = nltk.FreqDist(all_ngrams)

        self._top_k_grams = [x[0] for x in ngrams_freq.most_common(self._k)]


    def _extract_features(self, urls: Iterable[str], use_tqdm=True) -> list[list[int]]:

        if self._top_k_grams is None:
            raise ValueError("Top k ngrams not set, please call fit first")

        result = []

        for url in tqdm(urls, desc="Extracting features", disable=not use_tqdm):
            ngrams_in_url = nltk.FreqDist(nltk.ngrams(url, self._n))
            result.append(
                [ngrams_in_url.get(g, 0) for g in self._top_k_grams]
            )
            
        return result

    def fit(self, train_urls: list[str], train_labels: list) -> 'UrlClassifier':
        self._extract_top_k_grams(train_urls)
        train_features = self._extract_features(train_urls)
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
    

@dataclass
class Args:
    datafile: str
    n_samples: Optional[int]
    top_k_ngrams: int
    n_estimators: int
    ngram_size: int

def get_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datafile', type=str, required=True, help="csv file containing the data")
    parser.add_argument('-n', '--n_samples', type=int, default=None, help="number of samples to use after dropping na")
    parser.add_argument('-k', '--top_k_ngrams', type=int, default=500, help="number of ngrams to use")
    parser.add_argument('--n-estimators', type=int, default=500, help="number of estimators for the classifier")
    parser.add_argument('--ngram-size', type=int, default=2, help="ngram size for the classifier")

    args = parser.parse_args()

    return Args(**vars(args))


def train_and_classify(train_data: list[str], targets: list, test_feats: list[str], top_k_ngrams, n_estimators, ngram_size):
    # clf = RandomForestClassifier(n_estimators=500, random_state=42)
    # clf = GradientBoostingClassifier(n_estimators=500, random_state=42)
    clf = UrlClassifier(top_k_ngrams=top_k_ngrams, n_estimators=n_estimators, ngram_size=ngram_size)
    clf.fit(train_data, targets)
    return clf.predict(test_feats)


@dataclass
class EvaluateResult:
    precision: float
    recall: float
    f1: float

def evaluate(predictions, targets):
    # evaluate based on precision, recall and fscore
    precision = precision_score(targets, predictions, average='macro')
    recall = recall_score(targets, predictions, average='macro')
    f1 = f1_score(targets, predictions, average='macro')

    return EvaluateResult(precision, recall, f1)

def main(args: Args):

    # log args to mlflow
    mlflow.log_params(vars(args))

    data = pd.read_csv(args.datafile)
    
    if args.n_samples is not None:
        data = data.sample(n=args.n_samples, random_state=42)

    features_strs = data["url"].to_list()
    labels = data["is_dutch"].to_list()

    n_samples = len(features_strs)

    urls_train, urls_test, labels_train, labels_test = train_test_split(features_strs, labels, test_size=0.2, random_state=42)

    # classification using all features

    # predictions = train_and_classify(urls_train, labels_train, urls_test, args.top_k_ngrams)
    # evaluation = evaluate(predictions, labels_test)

    # Compare results for different values of k

    results = defaultdict(list)

    for top_k_ngrams in [100, 200, 300, 400, 500]:
        print(f"Top k ngrams: {top_k_ngrams}")
        predictions = train_and_classify(urls_train, labels_train, urls_test, top_k_ngrams, args.n_estimators, args.ngram_size)
        evaluation = evaluate(predictions, labels_test)

        results["top_k_ngrams"].append(top_k_ngrams)
        results["metric"].append("precision")
        results["value"].append(evaluation.precision)

        results["top_k_ngrams"].append(top_k_ngrams)
        results["metric"].append("recall")
        results["value"].append(evaluation.recall)

        results["top_k_ngrams"].append(top_k_ngrams)
        results["metric"].append("f1")
        results["value"].append(evaluation.f1)

    results = pd.DataFrame(results)

    fig = px.line(results, x="top_k_ngrams", y="value", color="metric", title=f"Performance of different top k ngrams, for {n_samples} samples")
    mlflow.log_figure(fig, "results_k_ngrams.html")

    # Compare results for different values of estimators

    results = defaultdict(list)

    for n_estimators in [25, 50, 75, 100, 200, 300, 400, 500]:
        print(f"N estimators: {n_estimators}")
        predictions = train_and_classify(urls_train, labels_train, urls_test, args.top_k_ngrams, n_estimators, args.ngram_size)
        evaluation = evaluate(predictions, labels_test)

        results["n_estimators"].append(n_estimators)
        results["metric"].append("precision")
        results["value"].append(evaluation.precision)

        results["n_estimators"].append(n_estimators)
        results["metric"].append("recall")
        results["value"].append(evaluation.recall)

        results["n_estimators"].append(n_estimators)
        results["metric"].append("f1")
        results["value"].append(evaluation.f1)

    results = pd.DataFrame(results)

    fig = px.line(results, x="n_estimators", y="value", color="metric", title=f"Performance of different n estimators, for {n_samples} samples")

    mlflow.log_figure(fig, "results_estimators.html")
    # for func in (mlflow.log_metric, print):
    #     func("precision_all_feats", evaluation.precision)
    #     func("recall_all_feats", evaluation.recall)
    #     func("f1_all_feats", evaluation.f1)


if __name__ == '__main__':
    with mlflow.start_run():
        main(get_args())