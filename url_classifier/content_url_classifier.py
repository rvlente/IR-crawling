import argparse
from dataclasses import dataclass
from typing import Optional
from nltk.util import ngrams
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import nltk
import mlflow
from tqdm import tqdm
import joblib
import numpy as np

memory = joblib.Memory(location="./cache/joblib_mem", verbose=0)

def gram2(s: str):
    return nltk.ngrams(s, 2)

# @memory.cache
def extract_features(strs_train: list[tuple[str, str]], strs_test: list[tuple[str, str]], use_context=True, top_k_ngrams=200) -> tuple[np.ndarray, np.ndarray]:

    # train_urls = [x[0] for x in strs_train]
    # train_contexts = [x[1] for x in strs_train]

    # test_urls = [x[0] for x in strs_test]
    # test_contexts = [x[1] for x in strs_test]

    # url_vectorizer = CountVectorizer(analyzer=gram2).fit(train_urls)
    # context_vectorizer = CountVectorizer(analyzer=gram2).fit(train_contexts)

    # url_feats_train = url_vectorizer.transform(train_urls).toarray()
    # url_feats_test = url_vectorizer.transform(test_urls).toarray()

    # if use_context:
    #     context_feats_train = context_vectorizer.transform(train_contexts).toarray()
    #     context_feats_test = context_vectorizer.transform(test_contexts).toarray()

    #     feats_train = np.hstack((url_feats_train, context_feats_train))
    #     feats_test = np.hstack((url_feats_test, context_feats_test))

    #     return feats_train, feats_test
    
    # return url_feats_train, url_feats_test

    all_url_ngrams = []
    all_context_ngrams = []

    for url, context in tqdm(strs_train, desc="Collecting ngrams from train data"):
        all_url_ngrams.extend(nltk.ngrams(url, n=2))
        all_context_ngrams.extend(nltk.ngrams(context, n=2))

    top_k_url_ngrams = [x[0] for x in nltk.FreqDist(all_url_ngrams).most_common(top_k_ngrams)]
    top_k_context_ngrams = [x[0] for x in nltk.FreqDist(all_context_ngrams).most_common(top_k_ngrams)]

    train_feats = []
    test_feats = []

    for i, (feats, data) in enumerate(zip([train_feats, test_feats], [strs_train, strs_test])):

        for url, context in tqdm(data, desc=f"Extracting features {i+1}/2"):
            url_ngrams_freq = nltk.FreqDist(nltk.ngrams(url, n=2))
            context_ngrams_freq = nltk.FreqDist(nltk.ngrams(context, n=2))

            url_ngrams_feats = [url_ngrams_freq.get(ngram, 0) for ngram in top_k_url_ngrams]

            if use_context:
                context_ngram_feats = [context_ngrams_freq.get(ngram, 0) for ngram in top_k_context_ngrams]
            else:
                context_ngram_feats = []

            feats.append(
                url_ngrams_feats + context_ngram_feats
            )
    
    return train_feats, test_feats
    

@dataclass
class Args:
    datafile: str
    n_samples: Optional[int]
    top_k_ngrams: int

def get_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datafile', type=str, required=True, help="csv file containing the data")
    parser.add_argument('-n', '--n_samples', type=int, default=None, help="number of samples to use after dropping na")
    parser.add_argument('-k', '--top_k_ngrams', type=int, default=200, help="number of ngrams to use")
    args = parser.parse_args()

    return Args(**vars(args))


def train_and_classify(train_feats, targets, test_feats):
    # clf = RandomForestClassifier(n_estimators=500, random_state=42)
    clf = GradientBoostingClassifier(n_estimators=500, random_state=42)
    clf.fit(train_feats, targets)
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
    data = data.dropna()
    
    if args.n_samples is not None:
        data = data.sample(n=args.n_samples, random_state=42)

    features_strs = list(zip(data["url"].to_list(), data["context"].to_list()))
    labels = data["is_dutch"].to_list()

    strs_train, strs_test, labels_train, labels_test = train_test_split(features_strs, labels, test_size=0.2, random_state=42)


    # classification using all features
    feats_train, feats_test = extract_features(strs_train, strs_test, use_context=True, top_k_ngrams=args.top_k_ngrams)

    predictions = train_and_classify(feats_train, labels_train, feats_test)
    evaluation = evaluate(predictions, labels_test)

    for func in (mlflow.log_metric, print):
        func("precision_all_feats", evaluation.precision)
        func("recall_all_feats", evaluation.recall)
        func("f1_all_feats", evaluation.f1)


    # classification using only url features
    feats_train, feats_test = extract_features(strs_train, strs_test, use_context=False, top_k_ngrams=args.top_k_ngrams)

    predictions = train_and_classify(feats_train, labels_train, feats_test)
    evaluation = evaluate(predictions, labels_test)

    for func in (mlflow.log_metric, print):
        func("precision_url_feats", evaluation.precision)
        func("recall_url_feats", evaluation.recall)
        func("f1_url_feats", evaluation.f1)


if __name__ == '__main__':
    with mlflow.start_run():
        main(get_args())