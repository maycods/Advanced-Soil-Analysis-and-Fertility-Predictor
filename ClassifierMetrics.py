import numpy as np

class ClassifierMetrics:
    """
    Class for computing classifier evaluation metrics.

    Attributes:
    - Y_test: The true labels.
    - y_pred: The predicted labels.

    Methods:
    - confusion_matrix(y_test, y_pred): Compute the confusion matrix.
    - Values(m): Extract TP, FN, FP, TN from the confusion matrix.
    - recall_score(TP, FN): Compute the recall score.
    - precision_score(TP, FP): Compute the precision score.
    - FP_rate(FP, TN): Compute the false positive rate.
    - specificity_score(TN, FP): Compute the specificity score.
    - accuracy_score(m): Compute the accuracy score.
    - f1_score(TP, FP, FN): Compute the F1 score.
    """

    def __init__(self, Y_test, y_pred):
        self.Y_test = Y_test
        self.y_pred = y_pred
    
    def confusion_matrix(self, y_test, y_pred):
        N = len(np.unique(y_test)) 
        M= np.zeros((N,N),dtype=int)
        for i in range(0,y_test.shape[0]) : 
            M[int(y_test[i])][int(y_pred[i])] += 1    
        return M

    def Values(self, m):
        TP= m.diagonal()
        FP = m.sum(axis=0) - TP
        FN = m.sum(axis=1) - TP
        TN =  m.sum() - (TP + FN + FP)
        return TP, FN, FP, TN
    
    def recall_score(self, TP, FN):
        return TP/(TP+FN)
    
    def precision_score(self, TP, FP):
        return TP/(TP+FP)
    
    def FP_rate(self, FP, TN):
        return  FP/(FP+TN)
    
    def specificity_score(self, TN, FP):
        return TN/(TN+FP)
    
    def accuracy_score(self, m):
        return np.sum(m.diagonal())/np.sum(m)
    
    def f1_score(self, TP, FP, FN):
        if any(self.recall_score(TP, FN)+self.precision_score(TP, FP))==np.nan:
            return 0
        return 2*(self.recall_score(TP, FN)*self.precision_score(TP, FP))/(self.recall_score(TP, FN)+self.precision_score(TP, FP))
