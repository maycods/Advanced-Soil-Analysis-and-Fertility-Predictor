import numpy as np
import statistics
import distance

class KNN:
    """
    Class for K-Nearest Neighbors classifier.

    Attributes:
    - k: The number of nearest neighbors to consider.
    - methode: The method used to calculate distances.

    Methods:
    - fit(xt, yt): Fit the model with training data.
    - _predict(Xtest): Predict labels for test data.
    """
    
    def __init__(self, k, methode) -> None:
        self.k = k
        self.methode = methode

    def fit(self, xt, yt):
        self.Xtrain = xt
        self.Ytrain = yt

    def _predict(self, Xtest):
        # Calculate Distances
        dist = np.apply_along_axis(lambda x: distance.distance(x, Xtest, self.methode), axis=1, arr=self.Xtrain)

        # Sort Distances
        ind = np.argsort(dist)

        # Select K Nearest Neighbors
        knn = self.Ytrain[ind[:self.k]]

        # Majority Voting
        Y = statistics.mode(knn)
        
        return Y