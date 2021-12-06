from url_classifier import UrlClassifier
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

        if results[0] > results[1] or results[0] > results[2]:
            print("Unexpected predictions")