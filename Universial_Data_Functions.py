# Alexander Teeuwen 06-03-2025
#------------------------------------------------------------------
# This file has a class with with methods to:
# Connect to the SQL database
# Create a pandas dataframe
# Several methods for data cleaning and understanding
# Method for the creation of new data
# Methods for visualizing data frames
#------------------------------------------------------------------
#Import the librarys
import pandas as pd
import glob
import time
import pyodbc
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import random
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from dataclasses import dataclass
from scipy.interpolate import UnivariateSpline
#---------------------------------
#Global vars
Connection_With_DB = False
#---------------------------------
#Data class to store a data reqeust
@dataclass
class SQL_Request:
    Table_Name: str
    Columns:    [] # type: ignore
    Names:      [] # type: ignore
    
#Data class to store artificcial data engineering requests (for adding extra data)
@dataclass
class Data_Add:
    column: str
    add: float = 0.0
#---------------------------------
#Connection data for the SQL server
server = 'sql_server,1433'
database = '822Raw'
username = 'SA'
password = 'Put password here'

driver='/usr/local/lib/libmsodbcsql.13.dylib'
#---------------------------------
# Make a connection to the SQL server
cnxn = pyodbc.connect(
'DRIVER={ODBC Driver 18 for SQL Server};'
'SERVER=' + server + ';'
'DATABASE=' + database + ';'
'UID=' + username + ';'
'PWD=' + password + ';'
'TrustServerCertificate=Yes'
)
cursor = cnxn.cursor()
Connection_With_DB = True
#---------------------------------   
#General functions
# Function for rmoving a digit
def remove_digit(string_in):
    no_digits = []
    for i in string_in:
        if i.isdigit() == False:
            no_digits.append(i)
    return ''.join(no_digits)   

# Function to check if a string contains only numbers
def is_numeric(s):
    return s.isdigit()
#---------------------------------   
#Class for building up a pandads dataframe and intercating with it
class SQL_Collecter:
    #Init the object
    def __init__(self,sql_handler,timestamp_begin,minutes_amount,database_name,ra_old = False):
        self.sql_handler = sql_handler
        self.timestamp_begin = timestamp_begin
        self.minutes_amount = str(minutes_amount)
        self.db_name = database_name
        self.ra_old = ra_old
        self.req_list =[]
        self.df = None
        #List for storing the data
        self.data_frame = None        
        self.splits = ['[',']']
        self.translation_dicts = []
        #Dictionarys for translating the tagset to an SQL table
        self.translation_dictL1 = {'AC':'AC'}
        self.translation_dictL2 = {'AH':'','FS':'_FS1_'}
        self.translation_dictL3 = {'AH':'HUM', 'AI':'SENSF','AS':'SENSS','AR': 'SENSR','AV':'AV', 'AM':'SENSM'}
        self.translation_dictL4 = {'SHU':'OUT', 'ATT':'AT','ART':'AH','FSC': 'MTR1ACT_SPD'}
        self.Specials = ['AC[]','AP[]','RA[]','AA[]', 'FM[]','IV_PS_FINAL','IV_PV_FLOW_TOTAL','IV_PS_FLOW_TOTAL','IV_CB_ECO']
        self.Multi_indexes = ['RA']
        self.Multi_indexes_SQL = ['RA1','R1']
    
        #Select between the old and the new style 
        if self.ra_old :
            translate_ra = 'AC01_R1TEMP'
        else: 
            translate_ra = 'AC01_RA1TEMP'
        #Build up the special  (data that can not be solved through simple logic)
        self.Translate_Specials = {
        'AC[1].AH[1].AC[1].WCV[1].IV_PV':'AC01CLVOUT',
        'AC[1].AH[1].AC[1].WTT[1].IV_PV_SCL' : 'AC01SENSCWIN',
        'AC[1].AH[1].AC[1].WTT[2].IV_PV_SCL' : 'AC01SENSCWOUT',
        'AC[1].AH[1].AA[1].WTT[1].IV_PV_SCL' : 'AC01SENSHTR_INLET',
        'AC[1].AH[1].AP[1].AHE[1].IV_PV' : 'AC01HTROUT',
        'AC[1].AH[1].AS[1].ATC[1].IV_PS_FINAL' : 'AC01SATSFINAL_SAT_SETP',       
        'AC[1].RA[1].CS[1].ATT[1].IV_PV_SCL' : translate_ra,
        'AC[1].RA[1].AS[1].ATT[1].IV_PV_SCL' : 'AC01_RA1TEMP_SUPPLY',
        'AC[1].AH[1].FM[1].FSC[1].IV_PV_SPD' : 'AC01_ACFANACT_SPD',
        'AC[1].AH[1].AS[1].IV_PV_FLOW_TOTAL' : 'AC01EXTRATOTAL_FLOW',
        'AC[1].AH[1].AS[1].IV_PS_FLOW_TOTAL' : 'AC01EXTRATOTAL_FLOW_SETP',
        'AC[1].AH[1].AS[1].ATC[1].IV_CB_ECO' : 'AC01MODEACTIVATE_ECOMODE'
        }
        #Create a list of dictionaries, so it is easier to itterate over them
        self.translation_dicts.append(self.translation_dictL1)
        self.translation_dicts.append(self.translation_dictL2)
        self.translation_dicts.append(self.translation_dictL3)
        self.translation_dicts.append(self.translation_dictL4)

    #This methods combines the methods the goal requests in merged pandas DF out
    def Create_DF(self,requests,fill_zero_room, fill_zero_other,AH_in_table = False):
        #Itterate throug the requests and create the request objects
        for i in requests:
            self.__Prepeare_SQL_Request__(tag = i,fill_zero_room= fill_zero_room, fill_zero_other= fill_zero_other,AH_in_table = AH_in_table)
        #Get the data from the SQL server and merge the dataframes into 1 pandas dataframe
        self.__Get_Data_And_Merge__()

    #Method for getting a part of the data (build up the query)
    def __get_part__(self,table_name,data, column_names):
        query = 'SELECT TOP('+self.minutes_amount+')[timestamp]'
        #Check if the data type is an list
        if type(column_names) == list:
            #Create the query
            for i in column_names:
                query += ',['+i+']'
        #Add the from part
        query += 'FROM [' + self.db_name + '].dbo.['+ table_name+ ']'
        #add the where part
        query += ' where [timestamp_utc] >= ' + self.timestamp_begin
        #Return the query
        self.sql_handler.execute(query)
        rows = self.sql_handler.fetchall()
        #Create a pandas dataframe from the SQL query
        df = pd.DataFrame.from_records(rows,columns=data)
        return df
    
    # This method gets the data from the SQL tables and merges these to 1 dataframe
    def __Get_Data_And_Merge__(self):
        df_stack = []
        #Itterate through the stack of request
        for i in self.req_list:
            #Get the raw data from the DB and append it to the list
            df_stack.append(self.__get_part__(table_name=i.Table_Name,data = i.Names,column_names= i.Columns))
        #Merge the dataframes together on the timestamp
        for i in df_stack:
            #Check if there is alread an itteration passed
            if type(self.df) == type(None):
                self.df = i
                self.df['Timestamp'] = self.df['Timestamp'].dt.round('min')
                #new_df = new_df.drop('Timestamp',axis = 1)
            #merge this df with the existing one
            else:
                try:
                    to_join = i
                    to_join['Timestamp'] = to_join['Timestamp'].dt.round('min')
                    #Drop the old timestamp
                    #to_join = to_join.drop('Timestamp',axis = 1)
                    self.df = self.df.merge(to_join, on = 'Timestamp', how = 'inner')                    
                except Exception as e:
                    print('DF join fault', e)
    
    #This method converts the tagname to a SQL table and column name
    def __Prepeare_SQL_Request__(self,tag,fill_zero_room, fill_zero_other,AH_in_table):
        exception = False
        #Check of the tag can be converted thorugh logic or it needs a special conversion
        if self.__Special_Check__(tag = tag):
            try:
                #Call the special conversion
                column_Name = self.__Special_Conversion__(tag=tag,fill_zero_other= fill_zero_other, fill_zero_room= fill_zero_room)
            except Exception as e:
                exception = True
                print("Colmun not is special list",e)
        else:
            try:
                #Call the logic conversion
                column_Name = self.__Logic_Conversion__(tag=tag,fill_zero_other= fill_zero_other, fill_zero_room= fill_zero_room,AH_in_table=AH_in_table)
            except:
                exception = True
                print("Failed to build up with logic")
        try:
            #Get the table where the data houses
            table = self.__Get_Table__(tag = tag)
        except:
            exception = True
            print("Table not found")

        if exception == False:
            #Check if the table is already in the request list
            found = False
            #Loop through the request lists
            for i in self.req_list:
                if i.Table_Name == table:
                    #Add the data to the column and name lists
                    i.Columns.append(column_Name)
                    i.Names.append(tag)
                    found = True
                    break
            #When the SQL request is not yet create for the specific table, create the SQL request
            if found == False:
                #Append a SQL request struct to the requests list
                self.req_list.append(SQL_Request(Table_Name= table, Columns=[column_Name],Names = ['Timestamp',tag]))

    #This method is used to detect if a tag can be converted by the use of logic or by a special conversion
    def __Special_Check__(self,tag):
        j = 1
        for i in tag.split('.'):
            if (remove_digit(i) in self.Specials) and j !=1:
                return True
            j += 1
        #When ther is no sepcial found return false
        return False
    
    #This method uses logic to convert a tag to a table name
    def __Logic_Conversion__(self,tag,fill_zero_room, fill_zero_other,AH_in_table):
        level = 1
        result = ''
        for i in tag.split('.'):
            for j in i.split('['):
                for k in j.split (']'):
                    #save the previous string
                    if k.isdigit():                      
                        trans_dict = self.translation_dicts[level -1].get(previous)
                        if len(trans_dict) != 0: 
                            if level ==1:
                                if ('RA' in tag and fill_zero_room) or (not('RA' in tag) and fill_zero_other):
                                    result = result + trans_dict + str(k).zfill(2)
                                else:
                                    result = result + trans_dict + str(k)
                            elif level ==3 and trans_dict == 'AV':
                                result = result + trans_dict + str(k) + '_'
                            else:
                                result = result + trans_dict
                        elif previous == 'AH' and AH_in_table:
                            result += 'AH1'
                        #Go 1 level deeper into the tag
                        level += 1
                    else:
                        previous = k
        result += '_VAL0'
        return result
    
    #This method converts the tag to a column with a more manual conversion
    def __Special_Conversion__(self,tag,fill_zero_room,fill_zero_other):
        tag_i = tag
        #Get the Installation number
        INS_Number = tag_i.split('.')[0].split('[')[1].split(']')[0]
        #Reset the L2 number
        L2_Number = 1
        #Replace the INS_Number with a 1 so it can be looked up in the dictonary
        to_replace = 'AC[' + INS_Number + ']'
        tag_i = tag_i.replace(to_replace,'AC[1]')
        #Check if the second level can have multiple indexes     
        multi_found = False   
        for i in self.Multi_indexes:
            if i in tag:                
                #A tag has been found that can have multiple indexes at level 2
                L2_Number = tag_i.split('.')[1].split('[')[1].split(']')[0]
                multi_found = True
                to_replace_L2 = i + '[' + L2_Number + ']'
                replace_to_L2 = i +'[1]'
                #Replace the index with index 1 so it can be looked up in the dictonary
                tag_i = tag_i.replace(to_replace_L2,replace_to_L2)
                break
        #Translate the tag to the column name
        result = self.Translate_Specials.get(tag_i)
        #Set the correct INS number
        if (('RA' in tag) and fill_zero_room) or ((not('RA' in tag)) and fill_zero_other):
            AC_Num = 'AC' + str(INS_Number).zfill(2)
        else:
            AC_Num = 'AC' + str(INS_Number)  .zfill(2)    
        result = result.replace('AC01',AC_Num).zfill(2)
        #If there is a level 2 multi index found, replace index 1 with the index that it should be
        if multi_found:
            succeeded = False
            for i in self.Multi_indexes_SQL:
                if i in result:
                    if fill_zero_room == False: 
                        L2 = remove_digit(i) + str(L2_Number)
                    else: 
                        L2 =  remove_digit(i) + str(L2_Number).zfill(2)
                    print(L2)
                    result = result.replace(i, L2)
                    succeeded = True
                    break
            #Throw an exception when the L2 conversion failed (Data not present in list)
            if succeeded == False:
                raise Exception("Level 2 tag not part of Multi_indexes_SQL") 
        result += '_VAL0'
        return result
    
    #This method is used to mask off certain parts of the data
    def mask_off(self,tag, value, minutes, greater_lower = 0):
        new_rows = []
        #Select what kind of masking needs to be done
        if greater_lower == 0:
            # Mask out the data where the value is equal
            mask_Data = self.df[self.df[tag] == value].copy()
        elif greater_lower == 1:
            #Mask out data that is greater
            mask_Data = self.df[self.df[tag] >= value].copy()
        elif greater_lower == 2:
            #Mask out data that is lower
            mask_Data = self.df[self.df[tag] <= value].copy()
        # Calculate the difference between consecutive timestamps
        mask_Data.loc[:, 'TimeDiff'] = mask_Data['Timestamp'].diff()
        # Identify the end of each block
        mask_Data.loc[:, 'EndOfBlock'] = mask_Data['TimeDiff'] > pd.Timedelta(minutes=1)
        # Mark the end timestamps
        mask_Data.loc[:, 'EndTimestamp'] = mask_Data['EndOfBlock'].shift(-1).fillna(False)
        # Filter to get only the end timestamps
        end_timestamps = mask_Data[mask_Data['EndTimestamp']]

        for idx, row in end_timestamps.iterrows():
            end_time = row['Timestamp']
            if idx + 1 < len(mask_Data):
                try:
                    next_time = self.df.loc[idx + 1, 'Timestamp'] if idx + 1 < len(mask_Data) else None
                    gap = (next_time - end_time).total_seconds() / 60 if next_time else minutes
                except:
                    print("Exception")
                    gap = minutes
            else:
                gap = minutes
            num_rows_to_add = min(minutes, int(gap))
            #print(num_rows_to_add,"rows to add")

            # Add new rows
            for i in range(1, num_rows_to_add + 1):
                new_time = end_time + pd.Timedelta(minutes=i)
                new_rows.append({'Timestamp': new_time})
        
        new_df = pd.DataFrame(new_rows)
  
        mask_Data2 = pd.concat([mask_Data, new_df]).sort_values(by='Timestamp').reset_index(drop=True)
        self.df = self.df[~self.df['Timestamp'].isin(mask_Data2['Timestamp'])]

    #This method gets the table where the data houses
    def __Get_Table__(self,tag):
        #Get the data from a air handling unit table
        if 'AH' in tag:
            #Check which airhandling unit table is required
            if 'TT' in tag or 'AHE' in tag or 'WCV' in tag or 'ATC' in tag:
                table = 'AHU_T'
            elif 'PT' in tag or 'FT' in tag or 'FLOW' in tag:
                table = 'AHU_P'
            elif 'RT' in tag or 'SHU' in tag:
                table = 'AHU_H'
            elif 'FM' in tag:
                #Get the index number
                INS_Number = tag.split('.')[0].split('[')[1].split(']')[0]
                #Build up the table
                table = 'AC' + str(INS_Number).zfill(2) + '_FANS'           
            else:
                print('no table found')
        #Get the data from a room table
        elif 'RA' in tag:
            #Get the index number
            INS_Number = tag.split('.')[0].split('[')[1].split(']')[0]
            #Build up the table
            table = 'AC' + str(INS_Number).zfill(2) + '_ROOMS'
        #Get the data from a non AC fan table
        elif 'FS' in tag:
            #Get data from a normal fan table
            if 'AV['in tag:
                #Get the index number
                INS_Number = tag.split('.')[0].split('[')[1].split(']')[0]
                #Build up the table
                table = 'AC' + str(INS_Number).zfill(2) + '_MTRS'   
        else:
            #Trigger an exception when no table has been found
            raise Exception("No corresponding table has been found") 
        return table
    
    #This method is used to detect if there is a NULL in the data
    def detect_null(self):
        rows_with_null = self.df.isnull().any(axis=1)
        rows_with_null.columns = ['Number','Is Null']

    #This method is used to filter out any non numeric rows of the dataframe
    def filter_na(self):        
        self.df = self.df.dropna()

    #This method is used to interpolate the gaps in the dataframe
    def interpolate(self,reset_index):
        self.df.interpolate(inplace = True)
        #Reset the index when this is requested
        if reset_index:
            self.df.reset_index(drop = True,inplace = True)

    #This method closes all the gaps in the dataframe and concats them with an intperolation
    def shorten_all_gaps(self,steps,reset_index):
        datablocks =[]
        end_found = False
        find_nan = True
        self.df.reset_index(inplace= True)
        #Keep looping till the end has been found
        while end_found == False:
            end_found = True
            #Loop through the complete dataframe in search of Nan rows
            for index, row in self.df.iterrows():
                has_nan = row.isnull().any()
                if (has_nan and find_nan) or (has_nan == False and find_nan == False):
                    #print(index)
                    if find_nan :
                        copy_part = self.df.iloc[:index].copy()                        
                        datablocks.append(copy_part)
                    #Remove this part from the origial dataframe
                    self.df = self.df.drop(self.df.index[:index]).reset_index(drop=True)
                    end_found = False
                    #Invert the find nan part
                    find_nan =  not find_nan
                    break
        result_df = pd.DataFrame()
        frame_empty = True
        #Combine the dataframes back into 1 big dataframe
        for i in datablocks:
            if frame_empty: 
                result_df = i
                result_df.drop('Timestamp',axis = 1,inplace = True)
                frame_empty = False
                print("Frame empty!!!!")
            else:               
                i.drop('Timestamp',axis = 1,inplace = True)
                result_df = self.__interpolate_and_concat__(i, steps = steps, df_External=result_df)
        self.df = result_df
        #reset the index (important for plotting and training a model)
        if reset_index:
            self.__Create_Timestamp__()

            self.df.reset_index(drop=True, inplace= True)
            self.df.drop(columns =['index'],inplace= True)

    #This method creates a moving avarge in the dataframe
    def moving_avarge_all(self,window): 
        #Get all the columns that are in the dataframe (except the timestamp)
        columns = self.__get_columns();
        #Loop throug the columns
        for i in columns:
            #Calculate the moving avarage for the desired columns
            self.df[i] = self.df[i].rolling(window=window, min_periods=1).mean()

    #This method is used to create a dataframe that is reudced and avarged
    def moving_avarage_red(self,window): 
        #Set the timestamp as the index before resampling
        self.df.set_index('Timestamp', inplace=True)       
        #resample the dataframe
        self.df = self.df.resample('15T').mean()

    #This method is used to detect an flatline in the data (for detecting a broken sensor)
    def flatline_detection(self,tag):
        #Get the lenght of the dataframe
        lenght = len(self.df)
        parts = lenght / 100
        print('Parts = ', parts)
        for i in range(int(parts)):
            if int(parts) * i <= lenght -100:
                start_value = self.df.loc[int(parts) * i,tag]
                #Check if all the other values are the same
                amount = 0
                for j in range(100):
                    k = 0
                    try:
                        if self.df.loc[(int(parts) * i) + j,tag] == start_value:
                            amount += 1                           
                        else:
                            amount = 0
                            start_value = self.df.loc[(int(parts) * i) + j,tag]
                            k =j
                    except:
                        pass
                if amount > 50:
                    self.__detect_jump__(tag = tag, position = (int(parts) * i) + k + amount)

    #This method is used to create a test dataframe to test the performance of a digital twin with a sine wave
    def create_twin_dest(self,start,lenght,tag,min,max): 
        #Get the original row
        original_row = self.df.iloc[start]
        #Create a sine wave
        x = np.linspace(0,2 * np.pi, lenght)
        sine_wave = (max - min) / 2 * np.sin(x) + (max + min) /2
        #Create the new dataframe with the sine wave
        new_rows = []
        for value in sine_wave:
            new_row = original_row.copy()
            new_row[tag] = value
            new_rows.append(new_row)
        #Create the new dataframe
        new_df = pd.DataFrame(new_rows)
        #Add the timestamp index to the digital twin
        new_df = self.__Create_Timestamp__(df_External= new_df)

        return new_df
    #Create a start position for the digital twin
    def Create_Twin_Start_Data(self,start,length):
        #Make a copy of a stating point
        row_copy =  self.df.iloc[start].copy()
        new_rows = []
        for i in range(0,length):
            new_rows.append(row_copy)
        #Create a dataframe from the new rows
        new_df = pd.DataFrame(new_rows)
        new_df = self.__Create_Timestamp__(df_External= new_df)
        return new_df

    #This method combines all methods for the extention of the dataframe
    def extend_df(self, start,length,to_change,interpolate_steps,timestamp_begin,gaussian = False,df_external = None,pyramid = False,df_copy = None): 
        #Create a list where the dataframes can be stored
        created_df = []
        df_return = None
        #loop thorugh all the data additions
        for data_arr in to_change:
            #Create the data
            created_df.append(self.__Create_Data__(start = start, lenght= length,gaussian= gaussian, change = data_arr,pyramid= pyramid,df_copy=df_copy))
        first = True
        #When the data is created the dataframe needs to be created
        for created_data in created_df:
            if first:
                df_return = self.__interpolate_and_concat__(df= created_data,steps= interpolate_steps,df_External=df_external)
            else:
                df_return = self.__interpolate_and_concat__(df= created_data,steps= interpolate_steps,df_External=df_return)
            first = False
        #When the process has been finished the timestamp needs to be added to the dataframe
        df_return = self.__Create_Timestamp__(Start_Time=timestamp_begin,df_External=df_return)
        #Return the created dataframe
        return df_return


    #This is a method that is used to create artificcial data to train on 
    def __Create_Data__(self,start ,lenght,change,gaussian = False,  pyramid = False,df_copy = None): 

        #Check if the dataframe is from an external source
        if df_copy is None:
            df_pointer = self.df
        else:
            df_pointer = df_copy
        #Set a fixed seed for the random generator (this gives the same outcome everytime)
        np.random.seed(1)  # Also set seed for NumPy random generator
        #Get the portion of data that is uses as a base for the creation of data
        portion_df = df_pointer.iloc[start:lenght].copy()
        #Drop the timesereis form the dataframe
        if 'Timestamp' in portion_df:
            portion_df.drop('Timestamp',axis = 1,inplace = True)
        #Loop through the array with cahnge information
        for item in change:
            #Check if a piramid needs to be generated from the data
            if pyramid:
                # Create a pyramid pattern
                half_size = portion_df.shape[0] // 2
                pyramid_d = np.concatenate([
                    np.linspace(0, item.add, half_size),
                    np.linspace(item.add, 0, portion_df.shape[0] - half_size)
                ])
            #Select the type of noise
            if gaussian == False: 
                #Create some noise on the data that is within the range of the amount that is being added to the data
                noise = np.random.uniform(0-abs(item.add)/100, abs(item.add)/100, size=portion_df.shape[0])
            else:
                mean = 0  # Mean of the Gaussian noise
                std_dev = abs(item.add) / 100  # Standard deviation of the Gaussian noise
                # Generate Gaussian noise
                noise = np.random.normal(mean, std_dev, size=portion_df.shape[0])
            if pyramid:
                # Add the pyramid pattern and noise to the column
                portion_df[item.column] += (pyramid_d + noise)
            else:
                portion_df[item.column] += (item.add + noise )
        #Return the protion
        return portion_df

    #This method is used to create a smooth transistion from 1 dataframe to the next
    def __interpolate_and_concat__(self,df, steps, df_External = None): 
        #Check if the dataframe is from an external source
        if df_External is None:
            df_pointer = self.df
        else:
            df_pointer = df_External
        #Drop the timestamp if this is present in the dataframe
        if 'Timestamp' in df_pointer:
            df_pointer.drop('Timestamp',axis = 1,inplace = True)
        #Ensure both dataframes have the same amount of columns
        if not df_pointer.columns.equals(df.columns):
            #Raise an execption
            raise ValueError("Dataframes must have the same amount of columns")
        #Generate the interpolated rows
        interpolated_rows = []
        for i in range(1,steps+1): 
            weight = i / (steps + 1)
            interploated_row = df_pointer.iloc[-1] * (1-weight) + df.iloc[0] * weight
            interpolated_rows.append(interploated_row)
        #Create a dataframe from the interploated rows
        interpolated_df = pd.DataFrame(interpolated_rows, columns= df_pointer.columns)
        #Concat the Dataframes
        result_df = pd.concat([df_pointer,interpolated_df,df], ignore_index = True)
        return result_df

    #This method creates a timeseries index 
    def __Create_Timestamp__(self,Start_Time= None,df_External = None): 
        #Check if there is an external dataframe present
        if df_External is None:
            df_pointer = self.df
        else:
            df_pointer = df_External
        #Check if a starttime has been filled in, otherwise take the start time from the class
        if Start_Time is None: 
            start = self.timestamp_begin
        else:
            start = Start_Time
        #Generate a data range with 1 minute frequency
        data_range = pd.date_range(start= start,periods= len(df_pointer), freq= 'T') 
        #Put the index to the dataframe
        df_pointer['Timestamp'] = data_range
        if df_External is None: 
            #Retun nothing when the classes dataframe is used
            return None
        else:
            return df_pointer

    #This method is used to detect if a value jumps after a flatline or that it grafually changes
    def __detect_jump__(self,tag,position):
        #Get the latest value
        flat_value = self.df.loc[position,tag]
        #Loop until the value changes 
        i = 1
        while flat_value == self.df.loc[position+ i,tag]:
            i+=1
        no_flat_value = self.df.loc[position+ i,tag]

        print( "value goes from",flat_value, "to", no_flat_value)

    #This method returns 2 dataframes with the min and the max values for a specific column
    def retun_min_max(self,column,df_External = None):
        #Check if the interal or the external dataframe is used
        if df_External is None:
            df_pointer = self.df
        else:
            df_pointer = df_External
        #Get the min and the max values
        minimal = df_pointer[column].min()
        maximal = df_pointer[column].max()
        #Get the rows that have this value
        minimal_rows = df_pointer[df_pointer[column]==minimal]
        maximal_rows = df_pointer[df_pointer[column]==maximal]
        #Get the first 2 rows from each dataframe
        retun_min = minimal_rows.iloc[0:1]
        retun_max = maximal_rows.iloc[0:1]
        #Return the dataframes
        return retun_min,retun_max

#------------------------------------------------------------------------------
#This class is an extension of SQL collector class, this class adds a vissualization part to the class
class Visualaize_Data(SQL_Collecter):
    def __init__(self, sql_handler, timestamp_begin,minutes_amount,database_name,ra_old = False):
        #call the parent method
        super().__init__(sql_handler, timestamp_begin,minutes_amount,database_name,ra_old=ra_old)
        plt.style.use('default')
        self.lower_range = 0
        self.upper_range = 100
        self.lower_range2 = 0
        self.upper_range2 = 100
        self.amount_of_Plots = 1
        self.inside = True

    #This is a method that reset the plots for a new plot (not being used anymore)
    def __reset_Plot(self):
        self.ax.tick_params(colors='#F8F8F2')
        self.ax.xaxis.label.set_color('#F8F8F2')
        self.ax.yaxis.label.set_color('#F8F8F2')
        self.ax.spines['bottom'].set_color('#6272A4')
        self.ax.spines['left'].set_color('#6272A4')
        self.ax.title.set_color('#F8F8F2')
        self.ax.set_ylim(self.lower_range, self.upper_range)
        self.style = ['b-', 'r-', 'y-', 'g-', 'c-', 'm-', 'r-']
        self.fig.clf()  # Clear the figure
        self.ax = self.fig.add_subplot(111)  # Recreate the axes
        plt.figure(figsize=(10,600))
        self.ax.clear()  # Clear the axes

    #This is a method that reset the plots for a new plot
    def __reset_plot_light(self,title):
        if self.amount_of_Plots == 1:
            self.fig, self.ax = plt.subplots(self.amount_of_Plots, 1,figsize=(15, 6))
        elif self.amount_of_Plots ==2:
            self.fig, (self.ax, self.ax2) = plt.subplots(self.amount_of_Plots,1 ,  sharex=True, gridspec_kw={ 'hspace': 0.02},figsize=(10, 6))
            self.ax.tick_params(axis='x', which='both', length=3)
            self.ax2.tick_params(axis='x', which='both', length=3)
        self.ax.set_title(title, color='#000000')  # You can customize the color as 

        self.ax.tick_params(colors='#000000')  # Black ticks
        self.ax.xaxis.label.set_color('#000000')  # Black x-axis label
        self.ax.yaxis.label.set_color('#000000')  # Black y-axis label
        self.ax.spines['bottom'].set_color('#000000')  # Black bottom spine
        self.ax.spines['left'].set_color('#000000')  # Black left spine
        self.ax.title.set_color('#000000')  # Black title
        plt.figure(figsize=(200,6))
        #Select the amount of sub plots
        if self.amount_of_Plots == 2:            
            self.ax2.set_ylim(self.lower_range2, self.upper_range2)
        self.style = ['r-', 'b-', 'y-', 'g-', 'c-', 'r-']  # Removed yellow line 
            
   #Plot the data from the dataframe
    def Plot_Data(self,title,legend_heigth,legend = True, custom_names = None,light_mode = False,columns_first_graph=None, columns_second_graph=None, y_title_1 = '', y_title_2 = ''):
        #Reset the plot 
        if light_mode == False:
            self.__reset_Plot()
        else:
            self.__reset_plot_light(title= title)
        #Set the y axis limits for the first sub plot
        self.ax.set_ylim([self.lower_range, self.upper_range])
        #Check if there is a translation dictornary to translate the column names to more readable names
        if custom_names == None:
            #Plot the normal dataframe without the changes names
            self.__Plot_Df__(df_in=self.df,legend=legend,height=legend_heigth,y_title_1=y_title_1, y_title_2=y_title_2)
        else:
            #Make a copy of the original dataframe
            copy = self.df.copy()
            #Change the column names
            copy.rename(columns = custom_names,inplace = True)
            #Check if there 
            if columns_second_graph is None:
                self.__Plot_Df__(df_in=copy,height=legend_heigth,y_title_1=y_title_1)
            else:
                df_first = copy[columns_first_graph].copy()
                df_second = copy[columns_second_graph].copy()
                self.__Plot_Df__(df_in=df_first,df_in_sec= df_second,legend=legend,height=legend_heigth,y_title_1=y_title_1, y_title_2=y_title_2)
            self.fig.tight_layout()  # Adjust layout 


    #This method is used to plot a custom dataframes
    def Plot_Custom(self,title,legend_heigth,df_in, start = 0, end=0,legend = True, custom_names = None,light_mode= False,columns_first_graph=None, columns_second_graph=None, y_title_1 = '', y_title_2 = ''):
        if light_mode == False:
            self.__reset_Plot()
        else:
            self.__reset_plot_light(title= title)
        self.ax.set_ylim([self.lower_range, self.upper_range])

        #Check if there is a special column section
        if custom_names == None:
            #Plot the normal dataframe without 
            self.__Plot_Df__(df_in=df_in,legend=legend,height=legend_heigth)
        else:
            #Make a copy of the original dataframe
            copy = df_in.copy()
            #Change the column names
            copy.rename(columns = custom_names,inplace = True)
            if columns_second_graph is None:
                self.__Plot_Df__(df_in=copy,height=legend_heigth,y_title_1=y_title_1)
            else:
                df_first = copy[columns_first_graph].copy()
                df_second = copy[columns_second_graph].copy()
                self.__Plot_Df__(df_in=df_first,df_in_sec= df_second,legend=legend,height=legend_heigth,y_title_1=y_title_1, y_title_2=y_title_2)

        self.fig.tight_layout()  # Adjust layout

     #Plot a dataframe, general code
    def __Plot_Df__(self,df_in,height, start = 0, end = 0,legend = True,df_in_sec = None,y_title_1 = '', y_title_2 = ''):
        plt.figure(figsize=(15,80))
        # Check if there is a dataframe
        if df_in is not None:  
            #Check if only a part of the dataframe needs to be shown
            if start == 0: 
                df_in.plot(ax=self.ax, style=self.style, x = 'Timestamp')
                if df_in_sec is not None:
                    df_in_sec.plot(ax=self.ax2, style=self.style, x = 'Timestamp')
                    if self.inside:
                        self.ax2.legend(loc='best', ncol=1,fontsize =10).set_visible(legend)
                    else:
                        plt.xticks(rotation=0, ha='center', va='top')  # Adjust horizontal and vertical alignment
                        
                        self.ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10,ncol=1).set_visible(legend)
                        self.ax2.set_xlabel('Timestamp', labelpad=-10)  # Increase the labelpad value to move
                    self.ax2.set_ylabel(y_title_2)
                    #self.ax2.xaxis.set_tick_params(pad=0.5)
            else:
                #Try to extract a subset
                try:
                    #Show only a poart of the dataframe
                    subset_df = df_in.iloc[start:end]
                    subset_df.plot(ax=self.ax, style=self.style, x = 'Timestamp')
                    
                except:
                    print("Indexes out of bounce")
            
            #self.ax.set_ylabel(df_in, color='#F8F8F2')
            self.ax.set_ylabel(y_title_1)
            if self.inside:
                self.ax.legend(loc='best', ncol=1, fontsize=10)
            else:
                self.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, ncol=1).set_visible(legend)
            #plt.setp(legend.get_texts(), color='#F8F8F2')  # Set legend text color to dracula foreground color
            plt.draw()  # Redraw the plot
            # Show the plot
            plt.show()
#------------------------------------------------------------------------------------------------------











