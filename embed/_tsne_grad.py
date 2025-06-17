import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold._t_sne import _joint_probabilities_nn, _kl_divergence_bh
MACHINE_EPSILON = np.finfo(np.double).eps



class TSNE_GRAD():

    def __init__(
        self,
        n_components=2,
        *,
        perplexity=30.0,
        early_exaggeration=12.0,
        min_grad_norm=1e-7,
        metric="euclidean",
        metric_params=None,
        init="pca",
        verbose=0,
        random_state=None,
        method="barnes_hut",
        angle=0.5,
        n_jobs=None,
        n_iter="deprecated",
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.metric_params = metric_params
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.angle = angle
        self.n_jobs = n_jobs
        self.n_iter = n_iter


    def get_grad(self, X:np.ndarray, pos):
        n_samples = X.shape[0]
        n_neighbors = min(n_samples - 1, int(3.0 * self.perplexity + 1))

        knn = NearestNeighbors(
            algorithm="auto",
            n_jobs=self.n_jobs,
            n_neighbors=n_neighbors,
            metric=self.metric,
            metric_params=self.metric_params,
        )
        knn.fit(X)
        distances_nn = knn.kneighbors_graph(mode="distance")
        del knn
        distances_nn.data **= 2

        P = _joint_probabilities_nn(distances_nn, self.perplexity, self.verbose)

        degrees_of_freedom = max(self.n_components - 1, 1)

        p = pos.copy().ravel()

        args = [P, degrees_of_freedom, n_samples, self.n_components]
        error, grad = _kl_divergence_bh(p, *args)

        grad = grad.reshape(n_samples, self.n_components)

        return grad


    