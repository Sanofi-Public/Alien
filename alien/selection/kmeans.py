import numpy as np
from sklearn.cluster import KMeans

from .selector import SampleSelector


class KmeansSelector(SampleSelector):
    """Selector based on K-Means algorithm."""

    def _select(self, samples=None, batch_size=None, **kwargs):
        cluster_learner = KMeans(n_clusters=batch_size)
        X = self.model.embedding(samples)
        cluster_learner.fit(X)
        cluster_idxs = cluster_learner.predict(X)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (X - centers) ** 2
        dis = dis.sum(axis=1)
        q_idxs = np.array(
            [
                np.arange(X.shape[0])[cluster_idxs == i][
                    dis[cluster_idxs == i].argmin()
                ]
                for i in range(batch_size)
            ]
        )
        return q_idxs
