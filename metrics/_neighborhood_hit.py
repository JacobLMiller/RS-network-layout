from .metric import ArrayMetric
from sklearn.neighbors import NearestNeighbors
import numpy as np

class NHit(ArrayMetric):
    """
    Finds neighborhood hit between two different vector spaces of equal sample size

    """
    def __init__(self, X, Y, X_is_dist, Y_is_dist):
       super().__init__(X, Y, X_is_dist, Y_is_dist, compute_dist=False)

    def compute(self, k):
        """
        Parameters
        ----------

        k : int
            Number of nearest neighbors
        """
        def nn_indices(arr, dist):
            if dist is not None:
                indices = np.argsort(dist, axis=1)[:, 1:k+1]
            else:
                nn = NearestNeighbors(n_neighbors=k+1).fit(arr)
                indices = nn.kneighbors(arr, return_distance=False)[:,1:]
            return indices
        

        x_indices = nn_indices(self.X, self.dX)
        y_indices = nn_indices(self.Y, self.dY)

        n_hits = (x_indices[:, :, np.newaxis] == y_indices[:, np.newaxis, :]).sum(axis=(1, 2))
        return n_hits.mean() / k