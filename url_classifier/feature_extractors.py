



class Cctld(url):
    def __init(self):
        self.prepared = True

    def prepare(self, train_urls: list[str]):
        return

    def extract_features(self):
        return [word for word in re.findall(r'[a-zA-Z]+', url) if word in ['nl', 'be', 'sr']]

class Extract_word_features(url):
    def __init(self):
        self.prepared = True

    def prepare(self, train_urls: list[str]):
        return

    def extract_features(self):
        return [word for word in re.findall(r'[a-zA-Z]+', url) if len(word) > 2 and word not in exlude_words]

class Extract_top_k_grams_size_n():
    def __init(self, _top_k_grams, n):
        self.prepared = False
        self._top_k_grams = _top_k_grams
        self._n = n

    def prepare(self, train_urls: list[str]):
        all_ngrams = (nltk.ngrams(url, self._n) for url in train_urls)
        all_ngrams = [item for sub_list in all_ngrams for item in sub_list]

        ngrams_freq = nltk.FreqDist(all_ngrams)

        self._top_k_grams = [x[0] for x in ngrams_freq.most_common(self._k)]

    def extract_features(self, urls: Iterable[str], parallel_feature_extraction=True):
        if self._top_k_grams is None:
            raise ValueError("Top k ngrams not set, please call fit first")

        def extract_fn(url: str, top_k_grams, n):
            ngrams_in_url = nltk.FreqDist(nltk.ngrams(url, n))
            return [ngrams_in_url.get(g, 0) for g in top_k_grams]

        if parallel_feature_extraction:
            try:
                n_cpus = multiprocessing.cpu_count()
                result = Parallel(n_jobs=n_cpus, batch_size=len(urls) // n_cpus)(
                    delayed(extract_fn)(url, self._top_k_grams, self._n) for url in urls)
            except ValueError:
                result = [extract_fn(url, self._top_k_grams, self._n) for url in urls]
        else:
            result = [extract_fn(url, self._top_k_grams, self._n) for url in urls]

        return result

class Extract_top_k_grams_size_many():
    def __init(self, _top_k_grams, n):
        self.prepared = False
        self._top_k_grams = _top_k_grams
        self._n = n

    def prepare(self, train_urls: list[str]):
        all_ngrams = (nltk.ngrams(url, self._n) for url in train_urls)
        all_ngrams = [item for sub_list in all_ngrams for item in sub_list]

        ngrams_freq = nltk.FreqDist(all_ngrams)

        self._top_k_grams = [x[0] for x in ngrams_freq.most_common(self._k)]

    def extract_features(self, urls: Iterable[str], parallel_feature_extraction=True):
        if self._top_k_grams is None:
            raise ValueError("Top k ngrams not set, please call fit first")

        def extract_fn(url: str, top_k_grams, n):
            ngrams_in_url = nltk.FreqDist(nltk.ngrams(url, n))
            return [ngrams_in_url.get(g, 0) for g in top_k_grams]

        if parallel_feature_extraction:
            try:
                n_cpus = multiprocessing.cpu_count()
                result = Parallel(n_jobs=n_cpus, batch_size=len(urls) // n_cpus)(
                    delayed(extract_fn)(url, self._top_k_grams, self._n) for url in urls)
            except ValueError:
                result = [extract_fn(url, self._top_k_grams, self._n) for url in urls]
        else:
            result = [extract_fn(url, self._top_k_grams, self._n) for url in urls]

        return result