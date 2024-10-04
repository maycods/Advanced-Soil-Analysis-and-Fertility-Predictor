import numpy as np 
import math

def distance(instance1, instance2, methode):
    # Cosine
    if methode==0: 
         return  1-((np.sum([instance1[i]*instance2[i] for i in range(0,len(instance1))]))/(math.sqrt(np.sum([i**2 for i in instance1]))*math.sqrt(np.sum([i**2 for i in instance2]))))
    # Minkowski
    else:
        return sum( np.abs(instance1-instance2)**methode)**(1/methode)