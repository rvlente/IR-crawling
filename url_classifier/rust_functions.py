from .url_classifier import UrlClassifier
from .context_classifier import ContextClassifier
import numpy as np
import matplotlib.pyplot as plt
import time

model_cache = dict()


classifier_types = ["gradient_boosting", "SVM", "ProbaLinearSVC"]
feature_types = ["top_k_ngrams_size_n", "top_k_ngrams_size_many"]


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

def test_url_models():
    """
    Tests all the model and feature combinations for the url classifier.
    """

    dataPath = 'url_classifier/url_data_with_context.parquet'
    nr_data_points = 1000

    precision = []
    recall = []
    fscore = []
    times= []
    for classifier_type in classifier_types:
        per_classifier_precision = []
        per_classifier_recall = []
        per_classifier_fscore = []
        per_classifier_time = []
        for feature_type in feature_types:
            print(classifier_type, feature_type)
            result = UrlClassifier(classifier_type=classifier_type, feature_type=feature_type).test(dataPath, take=nr_data_points)
            per_classifier_precision.append(round(result["precision"], 3))
            per_classifier_recall.append(round(result["recall"], 3))
            per_classifier_fscore.append(round(result["fscore"], 3))
            per_classifier_time.append(round(result["seconds_prediction"], 3))
        precision.append(per_classifier_precision)
        recall.append(per_classifier_recall)
        fscore.append(per_classifier_fscore)
        times.append(per_classifier_time)
    precision = np.array(precision)
    recall = np.array(recall)
    fscore = np.array(fscore)
    times = np.array(times)

    fig, ax = plt.subplots(2, 2)
    matrixPlot(ax[0][0], precision, "Precision")
    matrixPlot(ax[0][1], recall, "Recall")
    matrixPlot(ax[1][0], fscore, "F-score")
    matrixPlot(ax[1][1], times, "Runtimes sec/" + str(nr_data_points) + " urls")

    fig.tight_layout()
    plt.show()

def test_context_models():
    """
    Tests all the model and feature combinations for the context classifier.
    """

    dataPath = 'url_classifier/url_data_with_context.parquet'

    precision = []
    recall = []
    fscore = []
    for classifier_type in classifier_types:
        per_classifier_precision = []
        per_classifier_recall = []
        per_classifier_fscore = []
        for feature_type in feature_types:
            result = ContextClassifier(classifier_type=classifier_type, feature_type=feature_type).test(dataPath, take=1000)
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
    xlabels = []
    for label in feature_types:
        xlabels.append(label.replace("top_k_", "", 1).replace("_n"," 2").replace("_many", " [3,4,5,6,7]"))
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, fontsize=6)
    ax.set_yticks(np.arange(len(classifier_types)))
    ax.set_yticklabels(classifier_types, fontsize=6)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=-45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(classifier_types)):
        for j in range(len(feature_types)):
            text = ax.text(j, i, round(data[i, j], 3),
                           ha="center", va="center", color="w")

    ax.set_title(title, fontsize=7)


