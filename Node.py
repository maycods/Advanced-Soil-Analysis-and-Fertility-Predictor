import numpy as np

class Node():
    """ 
    Class representing a node in a decision tree.

    Attributes:
    - feature_index: The index of the feature used for splitting.
    - threshold: The threshold value for the feature split.
    - left: The left child node.
    - right: The right child node.
    - info_gain: The information gain achieved by the split.
    - value: The predicted value if the node is a leaf node.
    """

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        #desicion node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        # leaf node
        self.value = value