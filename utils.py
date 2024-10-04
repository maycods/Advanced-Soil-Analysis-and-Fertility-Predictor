import numpy as np
import re
import copy as cp

"""
Functions:
    - val_manquante(attribute, dataset): Identify indices of missing values in the given attribute.
    - calcul_mediane(attribute, dataset): Calculate the median of the given attribute.
    - tendance_centrales_homeMade(attribute, dataset): Calculate central tendencies (mean, median, mode) of the given attribute.
    - quartilles_homeMade(attribute, dataset): Calculate quartiles of the given attribute.
    - ecart_type_home_made(attribute, dataset): Calculate the standard deviation of the given attribute.
"""

def val_manquante(attribute, dataset):
    L=[]
    for i in range(0,len(dataset[:,attribute])):
        if not re.fullmatch(r"\d+\.(:?\d+)?", str(dataset[i, attribute])):
            L.append(i)
    return L

def calcul_mediane(attribute, dataset):
    datasetCurrated=np.delete(dataset[:,attribute], val_manquante(attribute, dataset))
    liste = cp.deepcopy(datasetCurrated)
    liste.sort()
    if liste.size % 2 !=0 :
    
        mediane=liste[((liste.size+1)//2) -1]
    else :
        mediane=(liste[(liste.size//2)-1]+liste[liste.size//2])/2
    return mediane

def tendance_centrales_homeMade(attribute, dataset):
    datasetCurrated=np.delete(dataset[:,attribute], val_manquante(attribute, dataset))
    moyenne2 = datasetCurrated.sum() / datasetCurrated.shape[0]
    mediane2 = calcul_mediane(attribute, dataset)
    unique_values, counts = np.unique(datasetCurrated, return_counts=True)
    Indicemax = np.where(counts == max(counts))[0]
    mode2=[unique_values[i] for i in Indicemax]
    return [moyenne2,mediane2,mode2]

def quartilles_homeMade(attribute, dataset):
    datasetCurrated=np.delete(dataset[:,attribute], val_manquante(attribute, dataset))
    liste = cp.deepcopy(datasetCurrated)
    liste.sort()
    q0=liste[0]
    q1=(liste[liste.size//4-1]+liste[liste.size//4]) /2
    q3=(liste[liste.size*3//4-1]+liste[liste.size*3//4]) /2
    q2=calcul_mediane(attribute, dataset)
    q4=liste[-1]
    return [q0,q1,q2,q3,q4]

def ecart_type_home_made(attribute, dataset):
    datasetCurrated=np.delete(dataset[:,attribute], val_manquante(attribute, dataset))
    mean = np.mean(datasetCurrated)
    ecarts = [(val - mean) ** 2 for val in datasetCurrated]
    variance = np.mean(ecarts) 
    return np.sqrt(variance)