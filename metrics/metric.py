from sklearn.metrics import pairwise_distances

class ArrayMetric():
    def __init__(self, X, Y, X_is_dist: bool, Y_is_dist: bool, compute_dist: bool):
        if X_is_dist:
            self.dX = X
            self.X = None
        else:
            self.X = X
            if compute_dist:
                self.dX = pairwise_distances(X)
            else:
                self.dX = None

        if Y_is_dist:
            self.dY = Y
            self.Y = None
        else:
            self.Y = Y
            if compute_dist:
                self.dY = pairwise_distances(Y)
            else:
                self.dY = None