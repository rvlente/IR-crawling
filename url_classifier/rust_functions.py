from .url_classifier import UrlClassifier
from .context_classifier import ContextClassifier
from .combined_classifier import CombinedClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

model_cache = dict()


classifier_types = [
    # "gradient_boosting",
    "SVM",
    "ProbaLinearSVC"
]


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
    nr_data_points = 5000

    precision = []
    recall = []
    fscore = []
    times = []
    for classifier_type in classifier_types:
        per_classifier_precision = []
        per_classifier_recall = []
        per_classifier_fscore = []
        per_classifier_time = []
        for feature_type in feature_types:
            print(classifier_type, feature_type)
            result = UrlClassifier(
                classifier_type=classifier_type, feature_type=feature_type).test(
                dataPath, take=nr_data_points)
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
            result = ContextClassifier(
                classifier_type=classifier_type, feature_type=feature_type).test(
                dataPath, take=5000)
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
    plt.show()  # This doesnt work and i have no idea why ~Benjamin


def test_combined_models():
    """
    Tests all the model and feature combinations for the combined classifier.
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
            result = CombinedClassifier(
                classifier_type=classifier_type, feature_type=feature_type).test(
                dataPath, take=5000)
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
    plt.show()  # This doesnt work and i have no idea why ~Benjamin


def test_aggregated_model(classifier_type, feature_type):
    """
    Tests a combined url + context classifier.
    """

    dataPath = 'url_classifier/url_data_with_context.parquet'
    take = 5000

    url_clf = UrlClassifier(classifier_type=classifier_type, feature_type=feature_type)
    url_result = url_clf.test(dataPath, take=take)

    context_clf = ContextClassifier(classifier_type=classifier_type, feature_type=feature_type)
    context_result = context_clf.test(dataPath, take=take)

    print('precision url_clf', url_result['precision'])
    print('recall url_clf', url_result['recall'])
    print('fscore url_clf', url_result['fscore'])
    print()

    print('precision context_clf', context_result['precision'])
    print('recall context_clf', context_result['recall'])
    print('fscore context_clf', context_result['fscore'])
    print()

    # Load data
    df = pd.read_parquet(dataPath, engine='pyarrow')
    urls = df["url"][:take]
    contexts = df["url_context"][:take]
    features = np.column_stack((url_clf.predict_proba(urls)[:, 1], context_clf.predict_proba(contexts)[:, 1]))
    labels = df["is_dutch"][:take]

    X_train, X_test, y_train, y_test = train_test_split(features, labels, shuffle=True, train_size=0.9)
    def cutoff(x): return x > 0.56

    # AVG
    y_pred_avg = cutoff(np.average(X_test, axis=1))
    print('precision_avg', precision_score(y_test, y_pred_avg, pos_label=True).item())
    print('recall_avg', recall_score(y_test, y_pred_avg, pos_label=True).item())
    print('fscore_avg', f1_score(y_test, y_pred_avg, pos_label=True).item())
    print()

    # MAX
    y_pred_max = cutoff(np.max(X_test, axis=1))
    print('precision_max', precision_score(y_test, y_pred_max, pos_label=True).item())
    print('recall_max', recall_score(y_test, y_pred_max, pos_label=True).item())
    print('fscore_max', f1_score(y_test, y_pred_max, pos_label=True).item())
    print()

    # SVM
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV

    aggregation_classifier = CalibratedClassifierCV(LinearSVC(max_iter=10000))
    aggregation_classifier.fit(X_train, y_train)

    y_pred_svm = aggregation_classifier.predict(X_test)
    print('precision_svm', precision_score(y_test, y_pred_svm, pos_label=True).item())
    print('recall_svm', recall_score(y_test, y_pred_svm, pos_label=True).item())
    print('fscore_svm', f1_score(y_test, y_pred_svm, pos_label=True).item())
    print()

    # custom
    from itertools import combinations

    results = []

    for a, b in combinations((i / 100 for i in range(1, 100)), 2):
        def custom(X):
            r = []
            for x in X:
                r.append(x[0] > a or x[1] > b)
            return np.array(r)

        y_pred_custom = custom(X_test)
        results.append((precision_score(y_test, y_pred_custom, pos_label=True).item(), recall_score(
            y_test, y_pred_custom, pos_label=True).item(), f1_score(y_test, y_pred_custom, pos_label=True).item(), a, b))

    result = max(results, key=lambda x: x[2])

    print(result[3], result[4])
    print(f'precision_custom', result[0])
    print(f'recall_custom', result[1])
    print(f'fscore_custom', result[2])


def matrixPlot(ax, data, title):
    import numpy as np
    import matplotlib.pyplot as plt
    im = ax.imshow(data)

    # Show all ticks and label them with the respective list entries
    xlabels = []
    for label in feature_types:
        xlabels.append(label.replace("top_k_", "", 1).replace("_n", " 2").replace("_many", " [3,4,5,6,7]"))
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
