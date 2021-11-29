import os
from url_classifier import UrlClassifier

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
from joblib import Parallel, delayed
import time
import multiprocessing
from tempfile import TemporaryDirectory

@dataclass
class Args:
    datafile: str
    n_samples: Optional[int]
    top_k_ngrams: int
    n_estimators: int
    ngram_size: int
    classifier_type: str
    exp_name: str
    run_name: Optional[str]
    no_evaluate: bool
    save_file: Optional[str]
    classifier_type: str

def get_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datafile', type=str, required=True, help="csv file containing the data")
    parser.add_argument('-n', '--n-samples', type=int, default=None, help="number of samples to use after dropping na")
    parser.add_argument('-k', '--top-k-ngrams', type=int, default=500, help="number of ngrams to use")
    parser.add_argument('--n-estimators', type=int, default=500, help="number of estimators for the classifier")
    parser.add_argument('--ngram-size', type=int, default=2, help="ngram size for the classifier")
    parser.add_argument('-c', '--classifier-type', type=str, default="gradient_boosting", help="classifier type to use")
    parser.add_argument('--exp-name', type=str, default="default", help="name of the experiment")
    parser.add_argument('--run-name', type=str, default=None, help="name of the run")
    parser.add_argument('--no-evaluate', action='store_true', help="don't evaluate the model")
    parser.add_argument('--save-file', type=str, default=None, help="file to save the model to")

    args = parser.parse_args()

    return Args(**vars(args))


def train_and_classify(
        train_data: list[str], 
        targets: list, 
        test_feats: list[str], 
        top_k_ngrams=500, 
        n_estimators=500, 
        ngram_size=2,
        clf_type="gradient_boosting",
    ) -> tuple[np.ndarray, UrlClassifier]:
    # clf = RandomForestClassifier(n_estimators=500, random_state=42)
    # clf = GradientBoostingClassifier(n_estimators=500, random_state=42)
    clf = UrlClassifier(top_k_ngrams=top_k_ngrams, n_estimators=n_estimators, ngram_size=ngram_size, classifier_type=clf_type)
    clf.fit(train_data, targets)
    return clf.predict(test_feats), clf


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

    
    mlflow.set_experiment(args.exp_name)

    # set run name
    if args.run_name is not None:
        mlflow.set_tag("run_name", args.run_name)

    # log args to mlflow
    mlflow.log_params(vars(args))

    data = pd.read_csv(args.datafile)
    
    if args.n_samples is not None:
        data = data.sample(n=args.n_samples, random_state=42)

    features_strs = data["url"].to_list()
    labels = data["is_dutch"].to_list()


    if args.no_evaluate:
        urls_train, labels_train = features_strs, labels
        urls_test, labels_test = features_strs, labels
    else:
        urls_train, urls_test, labels_train, labels_test = train_test_split(features_strs, labels, test_size=0.2, random_state=42)

    # train and classify
    predictions, model = train_and_classify(
        urls_train, 
        labels_train, 
        urls_test, 
        top_k_ngrams=args.top_k_ngrams, 
        n_estimators=args.n_estimators, 
        ngram_size=args.ngram_size,
        clf_type=args.classifier_type,
    )

    # evaluate
    if not args.no_evaluate:
        evaluation = evaluate(predictions, labels_test)

        # mlflow.log_figure(fig, "results_estimators.html")
        for func in (mlflow.log_metric, print):
            func("precision_all_feats", evaluation.precision)
            func("recall_all_feats", evaluation.recall)
            func("f1_all_feats", evaluation.f1)

    if args.save_file is not None:
        joblib.dump(model, args.save_file)

    with TemporaryDirectory() as tmpdir:
        p = os.path.join(tmpdir, "model.joblib")
        model.save(p)
        mlflow.log_artifact(p)


if __name__ == '__main__':
    # import sys
    # print(sys.argv)
    # with mlflow.start_run(run_name="single"):
    main(get_args())