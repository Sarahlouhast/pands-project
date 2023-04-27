#Author: Sarah Hastings
#Course: 22-23: HDip in Computing in Data Analytics
#Module: Programming and Scripting
#Lecturer: Andrew Beatty
#Pandsproject: Fisher’s Iris data set - research, documentation, code, analysis


#Firstly import pandas to allow for data analysis, manipulation
import pandas as pd

#read iris.data file into a DataFrame, allowed through use of pandas imported
df = pd.read_csv('iris.data', delimiter=',')

#note the iris.data file does not contain the necessary column names, these are added using the below
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# read iris.data file into a DataFrame with column names
df = pd.read_csv('iris.data', delimiter=',', names=col_names)


#checking the first few rows of the DataFrame to check if loaded correctly, note head will default to showing first 5 unless specify more, if you want to see the last 5 you can use tail()
#print(df.head())

#Get information about the dataset - number of roms, columns, names of columns, data type, this will allow check if there are any null values
print(df.info())

