from .metric import ArrayMetric
import numpy as np
from zadu.measures import stress

class SNS(ArrayMetric):
    def __init__(self, X, Y, X_is_dist, Y_is_dist):
        super().__init__(X, Y, X_is_dist, Y_is_dist, compute_dist=True)

    """
    Computes Scale Normalized Stress
    """
    def compute(self):
        D_low_triu = self.dY[np.triu_indices(self.dY.shape[0], k=1)]
        D_high_triu = self.dX[np.triu_indices(self.dX.shape[0], k=1)]
        alpha = np.sum(D_low_triu * D_high_triu) / np.sum(np.square(D_low_triu))
        return self.compute_normalized_stress(alpha)
    
    def compute_normalized_stress(self,alpha):  
        """
        Compute normalized stress between X and alpha*Y using zadu's stress measure.
        """
        stressScore = stress.measure(self.X,alpha * self.Y,(self.dX, alpha * self.dY))
        return stressScore['stress']