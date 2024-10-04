import numpy as np
import Node

class DtClassifier:
    """
    Decision Tree Classifier.

    Parameters:
    - min_samples_split: The minimum number of samples required to split an internal node.
    - max_depth: The maximum depth of the tree.
    - info_gain_method: The method used to calculate information gain, either 'Gini' or 'Entropy'.
    - n_features: The number of features to consider when looking for the best split. If None, it uses all features.

    Methods:
    - fit(X, Y): Build a decision tree classifier from the training set.
    - predict(X): Predict class labels for samples in X.

    Attributes:
    - root: The root node of the decision tree.
    """

    def __init__(self, min_samples_split, max_depth, info_gain_method, n_features=None):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.info_gain_method = info_gain_method

    def build_tree(self, dataset, curr_depth=0):
        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)

        # Pre-pruning
        if num_samples < self.min_samples_split or curr_depth == self.max_depth:
            leaf_value = self.calculate_leaf_value(Y)
            return Node.Node(value=leaf_value)

        # find the best split
        best_split = self.get_best_split(dataset, num_samples, num_features)

        # Pre-pruning
        if best_split is None or "info_gain" not in best_split or best_split["info_gain"] <= 0:
            leaf_value = self.calculate_leaf_value(Y)
            return Node.Node(value=leaf_value)


        left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)
        right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)

        # Post-pruning
        current_info_gain = best_split["info_gain"]
        leaf_info_gain = self.information_gain(Y, None, None, self.info_gain_method)
        if leaf_info_gain >= current_info_gain:
            leaf_value = self.calculate_leaf_value(Y)
            return Node.Node(value=leaf_value)

        return Node.Node(
            best_split["feature_index"],
            best_split["threshold"],
            left_subtree,
            right_subtree,
            best_split["info_gain"]
        )
    
    def get_best_split(self, dataset, num_samples, num_features):
    
        best_split = {}
        max_info_gain = -float("inf")
        
        if self.n_features is not None:
            feature_indices = np.random.choice(num_features, self.n_features, replace=False)
        else:
            feature_indices = range(num_features)
        
        for feature_index in feature_indices:
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]

                    curr_info_gain = self.information_gain(y, left_y, right_y, self.info_gain_method)

                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
      
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child, mode):
        if l_child is None or r_child is None:
            return 0

        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)

        if mode == "Gini":
            gain = self.gini_index(parent) - (weight_l * self.gini_index(l_child) + weight_r * self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))

        return gain

    
    def entropy(self, y):
        
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
        
    def calculate_leaf_value(self, Y):

        Y = list(Y)
        return max(Y, key=Y.count)
    
    def fit(self, X, Y):
        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)

    
    def predict(self, X):
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        if tree.value is not None:
            return tree.value

        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)