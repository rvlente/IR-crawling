from url_classifier.url_classifier import UrlClassifier


def train_url_classifier(output_path, data_path, split, take, classifier_kwargs):
    classifier = UrlClassifier(**classifier_kwargs)
    print('Training URL classifier...')
    results = classifier.test(data_path, split=split, take=take)
    print(results)
    classifier.save(output_path)
    print(f'Classifier saved at {output_path}')
