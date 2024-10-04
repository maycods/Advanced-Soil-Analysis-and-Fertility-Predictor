import numpy as np
import distance
import random


class K_MEANS:
    """
    K-Means Clustering Algorithm.

    Parameters:
    - k (int): Number of clusters.
    - methode_d (str): Method to calculate distance between instances.
    - methode_c (str): Method to select initial centroids ('Random' or 'Better picking').
    - max_iterations (int): Maximum number of iterations for the algorithm.
    - dataset (numpy.ndarray): Dataset to be clustered.

    Methods:
    - fit(xt): Fits the model to the input data.
    - centroid_selection(methode): Selects the initial centroids.
    - _cluster(): Performs clustering on the dataset.
    - _prediction(instance): Predicts the cluster for a given instance.
    """

    def __init__(self,k,methode_d,methode_c,max_iterations, dataset) -> None:
        self.k = k
        self.centroid=[]
        self.dataset_letiqu = np.hstack((dataset[:,:].copy(), -1*np.ones((dataset[:,:].shape[0], 1))))
        self.methode_c=methode_c
        self.methode_d=methode_d
        self.max_iterations=max_iterations

    def fit(self,xt):
        self.Xtrain=xt
    def centroid_selection(self,methode):
        if methode=="Random":#random sans prendre le meme
            self.centroid.extend(self.Xtrain[random.sample(range(self.Xtrain.shape[0]), self.k),:])   
        elif methode=="Better picking":#better picking
            self.centroid.append(list(self.Xtrain[np.random.choice(self.Xtrain.shape[0]),:]))
            dist =  np.apply_along_axis(lambda x: distance.distance(x, self.centroid[0], self.methode_d), axis=1, arr=self.Xtrain)
            ind = np.argsort(dist)
            for i in range(self.k,0,-1):
                self.centroid.append(list(self.Xtrain[ind[int((len(ind)/self.k)*i )-1],:]))
    
    def _cluster(self):#instance
        #choose centroid 
        self.centroid_selection(self.methode_c)  
        #boucle
        change=True
        nbr_iteration=0
        while(change):
        #distance
            for j in range(self.Xtrain.shape[0]):
                distances=[]
                for i in range(self.k):
                    distances.append(distance.distance(instance1= self.centroid[i], instance2= self.Xtrain[j,:] ,methode=self.methode_d))
                #affectation
                c =np.argmin(distances)
                self.dataset_letiqu[j,-1]=c
            #maj centroid
            oldcentroid=self.centroid.copy()
            for i in range(self.k):
                cluster=np.array([row[:-1] for row in self.dataset_letiqu if row[-1]==i])
                self.centroid[i]=np.array([np.average(cluster[:,j]) for j in range(cluster.shape[1])] )

            if np.linalg.norm(np.array(self.centroid) - np.array(oldcentroid)) < 0.0001 or nbr_iteration>self.max_iterations:
                change=False
            nbr_iteration+=1
        return self.dataset_letiqu
    #bonus
    def _prediction(self,instance):
        distances=[]
        for i in range(self.k):
            distances.append(distance.distance(self.centroid[i],instance,self.methode_d))
        return np.argmin(distances),np.array([row[:-1] for row in self.dataset_letiqu if row[-1]==np.argmin(distances)])
