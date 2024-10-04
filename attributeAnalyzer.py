import utils
import numpy as np
import math
import matplotlib.pyplot as plt 

class AttributeAnalyzer:
    
    """
    Class for analyzing attributes in a dataset.

    Attributes:
    - dataset (numpy.ndarray): The dataset array.
    - dataFrame (pandas.DataFrame): The DataFrame containing dataset columns.

    Methods:
    - Boite_a_moustache(attribute, boolen): Generate a box plot of the given attribute.
    - scatterplot(attribute, attribute2): Generate a scatter plot of two attributes.
    - histogramme(attribute): Generate a histogram of the given attribute.
    - attribute_infos(attribute, outliers, scatter_attribute): Generate various visualizations and statistics for the given attribute.
    """
     
    def __init__(self, dataset, dataFrame):
        self.dataFrame = dataFrame
        self.dataset = dataset
    
    def Boite_a_moustache(self, attribute, boolean):
        abberante=[]
        liste=[]
        q3=utils.quartilles_homeMade(attribute, self.dataset)[-2]
        q1=utils.quartilles_homeMade(attribute, self.dataset)[1]
        IQR=(utils.quartilles_homeMade(attribute, self.dataset)[-2]-utils.quartilles_homeMade(attribute, self.dataset)[1])*1.5
        datasetCurrated=np.delete(self.dataset[:,attribute], utils.val_manquante(attribute, self.dataset))

        for var in datasetCurrated:
            if (var <(q3+IQR) and var>(q1-IQR)):
                liste.append(var)
            else:
                abberante.append(var)  
        if boolean == "With Outliers":
            plt.boxplot(datasetCurrated)
        
        else:
            plt.boxplot(liste)

    def scatterplot(self, attribute, attribute2):
        plt.scatter(self.dataset[:,attribute],self.dataset[:,attribute2],marker ='p')
        plt.title(f'Scatter Plot of the attributes {self.dataFrame.columns.tolist()[attribute]} and {self.dataFrame.columns.tolist()[attribute2]}')
        plt.xlabel(f'{self.dataFrame.columns.tolist()[attribute]} Attribute values')
        plt.ylabel(f'{self.dataFrame.columns.tolist()[attribute2]} Attribute values')

    def histogramme(self, attribute):
        datasetCurrated=np.delete(self.dataset[:,attribute], utils.val_manquante(attribute, self.dataset))
        plt.hist(datasetCurrated, bins=math.ceil(1+(10/3)*np.log10(self.dataset.shape[0])),edgecolor='black')
        plt.title(f'Histograme of the attribute {self.dataFrame.columns.tolist()[attribute]}')
        plt.xlabel('Attribute values')
        plt.ylabel('Frequence')
    
    def attribute_infos(self, attribute, outliers, scatter_attribute):
        moyenne2, mediane2, mode2 = utils.tendance_centrales_homeMade(attribute, self.dataset)
        q0, q1, q2, q3, q4 = utils.quartilles_homeMade(attribute, self.dataset)
        ecart_type = utils.ecart_type_home_made(attribute, self.dataset)

        hist_fig = plt.figure()
        self.histogramme(attribute)
        hist_fig.savefig("plots\\histogramme.png")
        plt.close(hist_fig)

        box_plot_fig = plt.figure()
        self.Boite_a_moustache(attribute, outliers)  
        box_plot_fig.savefig("plots\\boxplot.png")
        plt.close(box_plot_fig)

        scatter_plot_fig = plt.figure()
        self.scatterplot(attribute, scatter_attribute)
        scatter_plot_fig.savefig("plots\\scatterPlot.png")
        plt.close(scatter_plot_fig)

        plots = ["plots\\histogramme.png", "plots\\boxplot.png", "plots\\scatterPlot.png"]

        return moyenne2, mediane2, mode2, q0, q1, q2, q3, q4, ecart_type, plots
