import DtClassifier
import numpy as np


class RandomForestClassifier:
    """
    Random Forest Classifier.

    Parameters:
    - n_trees: int
        The number of decision trees in the random forest.
    - max_depth: int
        The maximum depth of each decision tree.
    - min_samples_split: int
        The minimum number of samples required to split an internal node.
    - n_features: int
        The number of features to consider when looking for the best split.
    - info_gain_method: str
        The method used to calculate information gain, either 'Gini' or 'Entropy'.

    Methods:
    - fit(X, Y): Fit the random forest classifier to the training data.
    - predict(X): Predict class labels for samples in X.

    Attributes:
    - trees: list
        List of decision tree classifiers in the random forest.
    """

    def __init__(self, n_trees, max_depth, min_samples_split, n_features, info_gain_method):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
        self.info_gain_method = info_gain_method

    def fit(self, X, Y):
        for i in range(self.n_trees):
            # Create a sub-dataset randomly
            subset_indices = np.random.choice(len(X), len(X), replace=True)
            subset_X = X[subset_indices, :]
            subset_Y = Y[subset_indices]

            # compute the decision tree of the sub-dataset
            tree = DtClassifier.DtClassifier(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                info_gain_method=self.info_gain_method,
                n_features=self.n_features
            )
            tree.fit(subset_X, subset_Y)

            self.trees.append(tree) #add it to the forest

    def predict(self, X):
        tree_predictions = [tree.predict(X) for tree in self.trees]

        #predict using majority voting 
        predictions = np.array(tree_predictions).T.astype(int)
        final_predictions = [np.argmax(np.bincount(prediction)) for prediction in predictions]
        return final_predictions
