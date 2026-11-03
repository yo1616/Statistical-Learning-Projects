import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer


class AutoBucketTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        k=100,
        ngram_range=(3, 5),
        max_features=150_000,
        sample_size=100_000,
        random_state=42,
    ):
        self.k = k
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.sample_size = sample_size
        self.random_state = random_state
        self.vec_ = None
        self.km_ = None

    def fit(self, inputs, y=None):
        _ = y

        features = np.asarray(inputs).astype(str)
        self.vec_ = TfidfVectorizer(
            analyzer="char",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            lowercase=True,
            sublinear_tf=True,
        )

        if self.sample_size and len(features) > self.sample_size:
            rng = np.random.default_rng(self.random_state)
            sample_indices = rng.choice(
                len(features), size=self.sample_size, replace=False
            )
            sampled_features = features[sample_indices]
        else:
            sampled_features = features

        vectors = self.vec_.fit_transform(sampled_features)
        self.km_ = MiniBatchKMeans(
            n_clusters=self.k,
            random_state=self.random_state,
            batch_size=4096,
            n_init="auto",
        ).fit(vectors)
        return self

    def transform(self, inputs):
        features = np.asarray(inputs).astype(str)
        vectors = self.vec_.transform(features)
        labels = self.km_.predict(vectors)
        row_indices = np.arange(len(labels))
        col_indices = labels
        data = np.ones_like(labels, dtype=np.float32)
        return sp.csr_matrix(
            (data, (row_indices, col_indices)), shape=(len(labels), self.k)
        )
