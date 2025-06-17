from .metric import ArrayMetric
import numpy as np
MAXIMUM_FLOAT = np.finfo(np.float64).max
from scipy.optimize import minimize_scalar

class SNKL(ArrayMetric):
    """
    Calculate Scale-Normalized KL Divergence (KL divergence and scale at the scale KL divergence is minimum)
    KL Divergence is defined as the loss function of the t-SNE algorithm
    """
    def __init__(self, X, Y, X_is_dist, Y_is_dist):
        super().__init__(X, Y, X_is_dist, Y_is_dist, compute_dist=True)

    def compute(self, perplexity=30, ret_scale=False, max_bound=MAXIMUM_FLOAT):
        """
        Parameters
        ----------

        perplexity : float
            Perplexity as described in Hinton and Roweis (2002) https://www.cs.toronto.edu/~hinton/absps/sne.pdf

        ret_scale : bool
            if True, returns scale, min_KL tuple
            else, returns min_KL

        max_bound : positive number
            NOTE: No longer relevant. Does not affect the result.
            upper bound of interval within which minimum is searched (If returned KL divergence is almost equal to max_bound, max_bound was likely too small)
        """
        # High-dimensional probability space
        P = self._joint_probabilities(perplexity)

        # Calculate KL divergence for scale=scale
        def get_kl(scale):
            Q = self._get_Q(scale=scale)
            kl_divergence = (P * np.log(P / Q)).sum()
            return kl_divergence
        
        # Calculate minimum KL
        # res = minimize_scalar(get_kl, bounds=(0, max_bound))
        res = minimize_scalar(get_kl)

        # if res.x > max_bound - 1:
        #     raise NotImplementedError(f"Finding minimum points at scales beyond max_bound (here {max_bound}) is not currently supported.")

        if ret_scale:
            return (abs(res.x), res.fun)
        else:
            return res.fun
            
        
    def _get_Q(self, scale):
        """
        Calculate the low dimensional probability distribution corresponding to Y
        Parameters
        ----------
        scale : float
          Scale to scale self.dY by
        """

        Q = (np.square(self.dY * scale) + 1.0) ** -1
        np.fill_diagonal(Q, 0)
        Q /= Q.sum()

        # To prevent zero division error
        Q = np.maximum(Q, 1e-15)

        return Q

    def _joint_probabilities(self, perplexity):
        n_samples = self.dX.shape[0]
        conditional_P = self._conditional_probabilities(perplexity)
        P = (conditional_P + conditional_P.T) / (2 * n_samples)
        P = np.maximum(P, 1e-15)
        return P

    def _conditional_probabilities(self, perplexity, steps=100):
        """
        Calculate conditional probability matrix P corresponding to SNE algorithm
        
        P[i, j] = P (j | i)
        
        Parameters
        ----------
        perplexity : Perplexity as described in Hinton and Roweis (2002) https://www.cs.toronto.edu/~hinton/absps/sne.pdf
        steps : Number of steps for binary search for variances of Gaussian distributions
        """
        n_samples = self.dX.shape[0]
        desired_entropy = np.full((n_samples, 1), np.log(perplexity))
        beta = np.ones((n_samples, 1)) # (2 * var_i ** 2) in the exponent of the Gaussian distribution
        beta_min = np.zeros((n_samples, 1), dtype=np.float64)
        beta_max = np.full((n_samples, 1), np.inf)

        # Binary Search
        for _ in range(steps):
            # Create conditional probability distributions
            P = np.exp(-1 * np.square(self.dX) / beta)
            np.fill_diagonal(P, 0)
            row_sums = P.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1e-8
            P = P / row_sums
            P = np.maximum(P, 1e-15)

            # Calculate entropy
            log_P = np.log(P)
            entropy = -1 * (P * log_P).sum(axis=1, keepdims=True)
            entropy_diff = entropy - desired_entropy


            # Stop if variances sufficiently match perplexity
            if np.all(np.abs(entropy_diff) < 1e-5):
                break
            

            # Binary search update:

            should_increase_beta = entropy_diff < 0.0

            # Update beta_min and beta_max
            beta_min = np.where(should_increase_beta, beta, beta_min)
            beta_max = np.where(should_increase_beta, beta_max, beta)

            beta_max_is_inf = (beta_max == np.inf)
        
            # Update beta

            mask = should_increase_beta & beta_max_is_inf
            beta[mask] = beta[mask] * 2

            mask = should_increase_beta & ~beta_max_is_inf
            beta[mask] = (beta[mask] + beta_max[mask]) / 2.0

            mask = ~should_increase_beta
            beta[mask] = (beta[mask] + beta_min[mask]) / 2.0

        return P
