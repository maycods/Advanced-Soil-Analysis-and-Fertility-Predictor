class Point:
    """
    Represents a point in a dataset for DBSCAN.

    Attributes:
    - instance: The coordinates of the point in the dataset.
    - marked: A boolean flag indicating whether the point has been visited or marked during clustering.
    - cluster: A boolean flag indicating whether the point belongs to a cluster.
    """

    def __init__(self, instance):
        self.instance=instance
        self.marked=False
        self.cluster=False
         