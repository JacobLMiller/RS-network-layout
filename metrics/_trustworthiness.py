from .metric import ArrayMetric
import numpy as np
from sklearn.manifold import trustworthiness

class SNS(ArrayMetric):
    def __init__(self, X, Y, X_is_dist, Y_is_dist):
        super().__init__(X, Y, X_is_dist, Y_is_dist, compute_dist=True)

    """
    Computes Trustworthiness
    """
    def compute(self,n_neighbors=5):
        if not self.Y: 
            print("low dimension must be given as coordinates")
            return -1.0

        if not self.X: return trustworthiness(self.dX,self.Y,metric="precomputed",n_neighbors=n_neighbors)
        return trustworthiness(self.X,self.Y,n_neighbors=n_neighbors)