import numpy as np
import re
import pandas as pd
import statistics
from datetime import datetime
from sklearn.linear_model import LinearRegression
import math
import utils 

class Preprocessing:
    """
    Class for preprocessing a dataset.

    Attributes:
    - dataset: The dataset to be preprocessed.
    - dataFrame: The pandas DataFrame representing the dataset.
    - numeric_columns: Index of numeric columns in the dataset.

    Methods:
    - val_manquante(attribute): Find indices of missing values in the given attribute.
    - calcul_mediane(attribute): Calculate the median of the given attribute.
    - quartilles_homeMade(attribute): Calculate quartiles of the given attribute.
    - ecart_type_home_made(attribut): Calculate the standard deviation of the given attribute.
    - Discretisation(attribute): Discretize the values of the given attribute.
    - remplacement_val_manquantes(method, attribute): Replace missing values in the given attribute.
    - remplacement_val_aberrantes(method, attribute): Replace outliers in the given attribute.
    - remplacement_manquant_generale(method): Replace missing values in all attributes.
    - remplacement_aberantes_generale(method): Replace outliers in all attributes.
    - normalisation(methode, attribute, vmin, vmax): Normalize the values of the given attribute.
    - normalisation_generale(methode, vmin, vmax): Normalize the values of all attributes.
    - reduire_row(): Remove duplicate rows from the dataset.
    - coef_correl(attribut1,attribut2): Calculate the correlation coefficient between two attributes.
    - reduire_dim(treashold): Reduce dimensions by removing attributes with high correlation.

    For dataset2:
    - year_mapping(time_period): Map a year to a time period.
    - convert_date(time_period, date): Convert a date to a standard format.
    - remplacement_val_manquantes2(method, attribut): Replace missing values in the given attribute.
    - remplacement_manquant_generale2(method): Replace missing values in all numeric attributes.
    - remplacement_aberantes_generale2(method): Replace outliers in all numeric attributes.
    """

    def __init__(self, dataset, dataFrame):
        self.dataset = dataset
        self.dataFrame = dataFrame   
        numeric_columns = self.dataFrame.select_dtypes(include=['int', 'float']).columns.tolist() # column label
        self.numeric_columns = [self.dataFrame.columns.get_loc(col) for col in numeric_columns]
  
    def Discretisation(self, attribute):
        vals = self.dataset[:,attribute].copy()
        vals.sort()
        q = 1+(10/3)*np.log10(self.dataset.shape[0])
        nbrelmt=math.ceil(self.dataset[:,attribute].shape[0]/q)
        
        for  val in range(0,self.dataset[:,attribute].shape[0]):  
            for i in range(0,vals.shape[0],nbrelmt):
                if(vals[i]>self.dataset[val,attribute]):
                    sup=i
                    break
            self.dataset[val,attribute]=np.median(vals[sup-nbrelmt:sup])
                
    def remplacement_val_manquantes(self, methode, attribute):
        missing=utils.val_manquante(attribute, self.dataset)
        for i in missing:
            if methode=='Mode':
                self.dataset[i,attribute]= statistics.mode(self.dataset[:,attribute])    
            else:
                self.dataset[i,attribute]= np.mean([self.dataset[j,attribute] for j in range(0,len(self.dataset)) if self.dataset[j,-1]==self.dataset[i,-1] and not j in missing])

    def remplacement_val_aberrantes(self, methode,attribute):
        abberante=[]
        if methode=='Linear Regression':
            IQR=(np.percentile(self.dataset[:, attribute], 75)-np.percentile(self.dataset[:, attribute], 25))*1.5
            for i in range(0,len(self.dataset[:,attribute])):
                if (self.dataset[i,attribute] >(np.percentile(self.dataset[:, attribute], 75)+IQR) or self.dataset[i,attribute]<(np.percentile(self.dataset[:, attribute], 25)-IQR)):
                    abberante.append(i)
            X = np.delete(self.dataset, attribute, axis=1)
            X = np.delete(X, abberante, axis=0)
            y=self.dataset[:,attribute]
            y= np.delete(y, abberante, axis=0).reshape(-1, 1)

            model = LinearRegression().fit(X, y)
            
            for i in abberante:
                x2=np.delete(self.dataset, attribute, axis=1)
                X_new =x2[i,:].T.reshape(1, -1)
                self.dataset[i,attribute]=model.predict(X_new)[0][0]
        elif methode=='Discritisation':
            self.Discretisation(attribute)
        else:
            IQR=(np.percentile(self.dataset[:, attribute], 75)-np.percentile(self.dataset[:, attribute], 25))*1.5
            for i in range(0,len(self.dataset[:,attribute])):
                if (self.dataset[i,attribute] >(utils.quartilles_homeMade(attribute, self.dataset)[-2]+IQR)) or (self.dataset[i,attribute]<(utils.quartilles_homeMade(attribute, self.dataset)[1]-IQR)):
                    if self.dataset[i,attribute]<np.percentile(self.dataset[:,attribute] ,20):
                        self.dataset[i,attribute] = np.percentile(self.dataset[:,attribute] ,20)
                    elif self.dataset[i,attribute]>np.percentile(self.dataset[:,attribute] ,80):
                        self.dataset[i,attribute] = np.percentile(self.dataset[:,attribute] ,80)
                    else:
                        self.dataset[i,attribute] =np.median(self.dataset[:,attribute])
        
    def remplacement_manquant_generale(self, methode):
        for i in range(0,self.dataset.shape[1]-1):
            self.remplacement_val_manquantes(methode,i) 

    def remplacement_aberantes_generale(self, methode):
        for i in range(0,self.dataset.shape[1]-1):
            self.remplacement_val_aberrantes(methode,i)
     
    def normalisation(self, methode, attribute, vmin, vmax):
        if methode=='Vmin-Vmax':
            vminOld=float(self.dataset[:,attribute].min())
            vmaxOld=float(self.dataset[:,attribute].max())
            for val in range(0,self.dataset[:,attribute].shape[0]):
                self.dataset[val,attribute]=vmin+(vmax-vmin)*((float(self.dataset[val,attribute])-vminOld)/(vmaxOld-vminOld))

        else:
            vmean=np.mean(self.dataset[:,attribute])
            s=np.mean( (self.dataset[:,attribute]  -vmean)**2)
            for  val in range(0,self.dataset[:,attribute].shape[0]):
                self.dataset[val,attribute]=(self.dataset[val,attribute]-vmean)/s 
    
    def normalisation_generale(self, methode, vmin, vmax):
        for i in range(0,self.dataset.shape[1]-1):
            self.normalisation(methode,i, vmin, vmax)

    def reduire_row(self):
        self.dataset= np.unique(self.dataset, axis=0, return_index=False)
    
    def coef_correl(self, attribut1,attribut2):
        moy1=np.mean(self.dataset[:,attribut1])
        moy2=np.mean(self.dataset[:,attribut2])
        e1=utils.ecart_type_home_made(attribut1, self.dataset)
        e2=utils.ecart_type_home_made(attribut2, self.dataset)
        return (self.dataset[:,attribut1].dot(self.dataset[:,attribut2])-(len(self.dataset)*moy1*moy2))/((len(self.dataset)-1)*(e1*e2))
    
    def reduire_dim(self, treashold):
        to_delete=[]
        for i in range(0,self.dataset.shape[1]-1):
            for j in range(i+1,self.dataset.shape[1]):
                if (np.abs(self.coef_correl(i,j))>treashold):
                    to_delete.append(i)
        self.dataset = np.delete(self.dataset,to_delete, axis=1)
        valid_indices = [col for col in to_delete if col < len(self.dataFrame.columns)]
        self.dataFrame = self.dataFrame.drop(self.dataFrame.columns[valid_indices], axis=1)


    #=======================================================DATASET2==========================================================
        
    def year_mapping(self, time_period):
        self.dataFrame['Start date'] = pd.to_datetime(self.dataFrame['Start date'], errors='coerce')
        self.dataFrame['end date'] = pd.to_datetime(self.dataFrame['end date'], errors='coerce')

        yearly_intervals = self.dataFrame.groupby((self.dataFrame['Start date'].dt.year))['time_period'].agg(['min', 'max'])

        year_mapping = {}

        for year, interval in yearly_intervals.iterrows():
            year_mapping[(interval['min'], interval['max'])] = int(year)

        for interval, y in year_mapping.items():
            if interval[0] <= int(time_period) <= interval[1]:
                return y
        
    def convert_date(self, time_period, date):
        date = str(date)
        dd_mm_yy = re.compile(r'\b\d{1,2}/\d{1,2}/\d{4}\b')
        dd_mmm = re.compile(r'\b\d{1,2}-[a-zA-Z]{3}\b')

        if dd_mm_yy.match(date):
            formatted_date = datetime.strptime(date, '%m/%d/%Y')
            return np.datetime64(formatted_date)
        elif dd_mmm.match(date):
            day, month = date.split('-')
            month_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
            month = month_dict[month]
            year = self.year_mapping(time_period)
            return np.datetime64(datetime(int(year), month, int(day)))
        else:
            return None
        
    def remplacement_val_manquantes2(self, method, attribut):
        missing = [i for i, val in enumerate(self.dataset[:, attribut]) if np.isnan(val)]
        
        for i in missing:
            zone = self.dataset[i, 0]
            time_period = self.dataset[i, 1]
            matching_rows = [z for z in range(self.dataset.shape[0]) if self.dataset[z, 1] == time_period and not np.isnan(self.dataset[z, attribut])]
            if method == "Mode":
                if matching_rows:
                    mode = statistics.mode(self.dataset[matching_rows, attribut])
                    self.dataset[i, attribut] = mode
                else:
                    zone_rows = [z for z in range(self.dataset.shape[0]) if self.dataset[z, 0] == zone and not np.isnan(self.dataset[z, attribut])]
                    mode = statistics.mode(self.dataset[zone_rows, attribut])
                    self.dataset[i, attribut] = mode
            else:
                if matching_rows:
                    mean_val = np.mean(self.dataset[matching_rows, attribut])
                    self.dataset[i, attribut] = mean_val
                else:
                    zone_rows = [z for z in range(self.dataset.shape[0]) if self.dataset[z, 0] == zone and not np.isnan(self.dataset[z, attribut])]
                    mean_val = np.mean(self.dataset[zone_rows, attribut])
                    self.dataset[i, attribut] = mean_val

    def remplacement_manquant_generale2(self, method):
        for attribute_index in self.numeric_columns:
           self.remplacement_val_manquantes2(method, attribute_index)
        
    def remplacement_aberantes_generale2(self, method):
        if method=="Discritisation":
            categorical_columns = [0, 1, 2]
            self.numeric_columns = [col for col in self.numeric_columns if col not in categorical_columns]
        for attribute_index in self.numeric_columns:
            self.remplacement_val_aberrantes(method, attribute_index)