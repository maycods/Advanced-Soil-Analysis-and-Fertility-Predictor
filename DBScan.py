import distance
import Point
import numpy as np

def Voisinage(P,radius,methode_d, dataset):
    """
    This function calculates the neighborhood of a given point P within a specified radius in a dataset, based on a given distance metric.
    """
    voisins=[]
    for i in range(dataset.shape[0]):
        if distance.distance(dataset[i,:],P.instance,methode_d) <=radius:
            voisins.append(i)
    return voisins

def DB_Scan(radius,min_points,methode_d, dataset):
    """
    This function implements the DBSCAN clustering algorithm. It iterates through each point in the dataset, identifying core points, expanding clusters, and marking outliers.
    """
    C = 0
    Outlier=[]
    dataset_labeled=[]
    listeP=[Point.Point(instance) for instance in dataset]
    
    for P in listeP:
        if not P.marked:
            P.marked=True
            PtsVoisins = Voisinage(P, radius,methode_d, dataset) 
            if len(PtsVoisins) < min_points :
                Outlier.append(P)
                dataset_labeled.append(np.append(P.instance,-1)) 
            else:
                C+=1 #new cluster
                P.instance=np.append(P.instance,C)
                P.cluster=True 
                dataset_labeled.append(P.instance)
                for i in PtsVoisins:
                    if not listeP[i].marked :
                        listeP[i].marked=True
                        v=Voisinage(listeP[i], radius,methode_d, dataset)
                        if len(v) >= min_points :
                            PtsVoisins.extend(v) 
                    if (not listeP[i].cluster) : 
                        listeP[i].cluster=True
                        if listeP[i] in Outlier:
                            Outlier.remove(listeP[i])
                            for j in range(len(dataset_labeled)):
                                if  np.array_equal( dataset_labeled[j][:-1],listeP[i].instance):
                                    listeP[i].instance=np.append(listeP[i].instance,C)
                                    dataset_labeled[j][-1]=C 
                                    break                        
                        else: 
                            listeP[i].instance=np.append(listeP[i].instance,C)
                            dataset_labeled.append(listeP[i].instance)
        
    return [list(i[:-1]) for i in dataset_labeled ],[i[-1] for i in dataset_labeled ],([i for i in dataset_labeled if i[-1]==-1])
