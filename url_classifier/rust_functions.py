from .url_classifier import UrlClassifier, classifier_types, feature_types
import pytest

model_cache = dict()


def predict_dutchiness_of_urls(urls: list[str], path_to_model: str) -> list[float]:
    """
    Classifies the urls and returns the classification results.
    """
    if len(urls) == 0:
        return []

    global model_cache
    classifier = model_cache.get(path_to_model, UrlClassifier.load(path_to_model))

    return classifier.predict_dutchiness(urls)


state = 0

def mutate_state():
    global state
    state += 1
    return state

    
def test_predict_dutchiness_of_urls():
    """
    Tests the predict_dutchiness_of_urls function.
    """
    import os

    urls = ["https://www.google.com", "https://www.google.nl", "https://www.google.com/nl"]
    path_to_model = "cache/model_full.joblib"


    if os.path.isfile(path_to_model):
        results = predict_dutchiness_of_urls(urls, path_to_model)
    else:
        classifier = UrlClassifier().fit(dataPath='url_classifier/url_data_with_context.parquet')
        classifier.save(path_to_model)
        results = classifier.predict_dutchiness(urls)

    if results[0] > results[1] or results[0] > results[2]:
        print("Unexpected predictions")
    print(results)

def test_all_models():
    """
    Tests aall the model and feature combinations
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import numpy as np

    dataPath = 'url_classifier/url_data_with_context.parquet'

    results = []
    ticks = []
    for classifier_type in classifier_types:
        for feature_type in feature_types:
            results.append(UrlClassifier(classifier_type=classifier_type, feature_type=feature_type).test(dataPath, take=1000))
            ticks.append(classifier_type + "_" + feature_type)

    precision = []
    recall = []
    fscore = []
    for result in results:
        precision.append(result["precision"])
        recall.append(result["recall"])
        fscore.append(result["fscore"])
    print(precision)

    plt.bar(np.arange(len(precision)), precision)
    plt.xticks(list(range(len(precision))), ticks)
    plt.show()
    plt.pause(10000)
