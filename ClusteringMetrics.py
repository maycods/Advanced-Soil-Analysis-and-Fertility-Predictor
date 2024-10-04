import distance
import numpy as np

class ClusteringMetrics:
    """
    Class for evaluating clustering results using different metrics : Silhouette Score, Inter Cluster Distance, Intra Cluster Distance.
    """

    def __init__(self, dataset, y_pred):
        self.dataset = dataset
        self.y_pred = y_pred
    
    def silhouette_score(self, data, labels, metric):
        num_points = len(data)
        unique_labels = np.unique(labels)
        silhouette_values = np.zeros(num_points)

        intra_cluster_distances = np.zeros(num_points)
        inter_cluster_distances = np.zeros(num_points)

        for i in range(num_points):
            #ai
            label_i = labels[i]
            cluster_i_indices = np.where(labels == label_i)[0] # get own cluster points
            if len(cluster_i_indices) == 1:
                silhouette_i = 0  # Set silhouette score to 0 for single point clusters
            else:
                a_i = np.mean([distance.distance(data[i],data[j],metric) for j in cluster_i_indices if j != i])
                inter_cluster_distances[i] = a_i

                #bi
                b_i_values = []
                for label_j in unique_labels:
                    if label_j != label_i:
                        cluster_j_indices = np.where(labels == label_j)[0] # get neighbor clusters points
                        b_ij = np.mean([distance.distance(data[i], data[j], metric) for j in cluster_j_indices])
                        b_i_values.append(b_ij)
                
                # get the average distance to the nearest neighbor cluster bi
                b_i = min(b_i_values) if b_i_values else 0
                intra_cluster_distances[i] = b_i

                # silhouette score of the point i
                silhouette_i = (b_i - a_i) / max(a_i, b_i)
                
            silhouette_values[i] = silhouette_i
                
        # silhouette score of data
        silhouette_score_avg = np.mean(silhouette_values)
        
        # Calculate overall intra-cluster and inter-cluster distances
        intra_distance = np.sum(intra_cluster_distances)
        inter_distance = np.sum(inter_cluster_distances)

        return silhouette_score_avg, intra_distance, inter_distance
