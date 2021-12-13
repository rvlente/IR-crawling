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
    Tests all the model and feature combinations
    """
    import numpy as np
    import matplotlib.pyplot as plt

    dataPath = 'url_classifier/url_data_with_context.parquet'

    precision = []
    recall = []
    fscore = []
    for classifier_type in classifier_types:
        per_classifier_precision = []
        per_classifier_recall = []
        per_classifier_fscore = []
        for feature_type in feature_types:
            result = UrlClassifier(classifier_type=classifier_type, feature_type=feature_type).test(dataPath, take=1000)
            per_classifier_precision.append(result["precision"])
            per_classifier_recall.append(result["recall"])
            per_classifier_fscore.append(result["fscore"])
        precision.append(per_classifier_precision)
        recall.append(per_classifier_recall)
        fscore.append(per_classifier_fscore)
    precision = np.array(precision)
    recall = np.array(recall)
    fscore = np.array(fscore)

    fig, ax = plt.subplots(3)
    matrixPlot(ax[0], precision, "precision")
    matrixPlot(ax[1], recall, "recall")
    matrixPlot(ax[2], fscore, "fscore")

    fig.tight_layout()
    plt.show() # This doesnt work and i have no idea why ~Benjamin

def matrixPlot(ax, data, title):
    import numpy as np
    import matplotlib.pyplot as plt
    im = ax.imshow(data)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(classifier_types)))#, labels=classifier_types)
    ax.set_yticks(np.arange(len(feature_types)))#, labels=feature_types)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(classifier_types)):
        for j in range(len(feature_types)):
            text = ax.text(j, i, f"{data[i, j]:.2f}",
                           ha="center", va="center", color="w")

    ax.set_title(title)


