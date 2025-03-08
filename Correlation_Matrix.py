#This file houses the code to calculate a covariance time series matrix
import pandas as pd
import seaborn as sns
import numpy as np

class Correlation_Matrix: 
    #Init method that is settings the dataframe
    def __init__(self, df_in,custom_names = None):
        #make a copy without the timestamp of the dataframe
        self.df = df_in.drop('Timestamp',axis= 1)     
        if custom_names != None:
            #Change the column names
            self.df.rename(columns = custom_names,inplace = True)    
        #Create an empty matrix to store the correlations
        self.columns = self.df.columns
        self.matrix = pd.DataFrame(index=self.df.columns, columns = self.columns)

    #This method creates a matrix with the correlations 
    def correlation_matrix(self,time,lag,plot_heatmap):
        #Loop through the columns
        for column_x in self.columns:
            #Loop thourg the columns a second time
            for column_y in self.columns:
                #Check if the columns are not the same
                if column_x != column_y:
                    #Calculate the correlation
                    self.matrix.loc[column_x,column_y] = round(self.__correlation__(column1= column_x, column2= column_y,lag =lag, time = time),2)
        #Fill in the remaining spots
        self.matrix = self.matrix.fillna(0)
        #plot the heatmap when desired
        if plot_heatmap: 
            self.__plot_heatmap__()

    #Method that is calculating the avarage of the window
    def __window_avarage__(self,column,point,lag): 
        #Check if there is enough data to calaulte the mean
        if (point - lag) >= 0:
            #Calculate the avarge
            return self.df.loc[point-lag:point+1][column].mean()
        else:
            return None
        
    #This method is used to calculate the covariance between 2 variables
    def __covariance__(self, column1,column2,lag,time):
        #Itterate thorugh all all the lags
        total_sum = 0
        #Calculate the window avarge
        avarge_i = self.__window_avarage__(column= column1,point = time, lag= lag)
        avarge_j = self.__window_avarage__(column=column2,point = time,lag= lag)
        #Check if the avarages are valid
        if avarge_i != None or avarge_j != None and lag != 0:  
            for i in range(lag +1 ):
                #Determine the index position
                pos = time - lag + i                            
                total_sum += (self.df[column1][pos] - avarge_i) * (self.df[column2][pos] - avarge_j)
                #Get the avarge in time
            return (lag ** -1) * total_sum
        else:
            return None
        
    #This method is calculating a sub part of the correlation question
    def __corrolation_sub__(self,column1,column2,lag,time,average_i,average_j):
        #Calculate the sum for i
        #Take the avarge of the window
        sum_i = 0
        #Sum the first column
        for i in range(lag +1):
            pos = time - lag + i  
            sum_i += ((self.df[column1][pos] - average_i) **2)
        #Calculate the sum for j
        #Take the avarge of the window
        sum_j = 0
        #Sum the second column
        for i in range(lag +1):
            pos = time - lag + i  
            sum_j += ((self.df[column2][pos] - average_j) ** 2)
        #Return the sub part
        return (sum_i * sum_j)
    
    #This method is used to calcualte the correlation between 2 varaibles
    def __correlation__(self,column1, column2,lag,time):
        #Check if the data input is correct
        avarge_i = self.__window_avarage__(column= column1,point = time, lag= lag)
        avarge_j = self.__window_avarage__(column=column2,point = time,lag= lag)
        if avarge_i != None or avarge_j != None and lag != 0: 
            #Calcualte the numerator of the equation
            numerator = 0
            #loop thorugh the window
            for i in range(lag +1 ):
                pos = time - lag + i  
                numerator += (self.df[column1][pos] - avarge_i) * (self.df[column2][pos] - avarge_j)
            #Calulate the denumerator
            denumerator = self.__corrolation_sub__(column1=column1,column2=column2,lag = lag, time=time,average_i=avarge_i,average_j=avarge_j)
            #Calculate the correlation
            if denumerator != 0:
                return numerator * (denumerator ** -0.5)
            else:
                return 0
        else:
            return None
    #This method plots a heatmap for the 
    def __plot_heatmap__(self):
        #Create a mask that only shows half a heatmap
        mask = np.triu(np.ones_like(self.matrix, dtype=bool))
        sns.heatmap(self.matrix, annot=True,mask=mask)
