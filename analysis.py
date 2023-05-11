#Author: Sarah Hastings
#Course: 22-23: HDip in Computing in Data Analytics
#Module: Programming and Scripting
#Lecturer: Andrew Beatty
#Pands project: Fisher’s Iris data set - research, documentation, code, analysis
#http://www.learningaboutelectronics.com/Articles/How-to-create-a-pairplot-Python-seaborn.php
#https://www.geeksforgeeks.org/box-plot-and-histogram-exploration-on-iris-data/


#Firstly import pandas to allow for data analysis, manipulation
import pandas as pd
#Use Pyplot, a submodule of the Matplotlib library to visualize the diagram/use for visual data/plots etc 
import matplotlib.pyplot as plt
#working with numerical data, arrays
import numpy as np
#seaborn give variety of visualization patterns in addition to matplotlib
import seaborn as sns
#read iris.data file into a DataFrame, allowed through use of pandas imported
df = pd.read_csv('iris.data', delimiter=',')

#note the iris.data file does not contain the necessary column names, these are added using the below
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

#read iris.data file into a DataFrame with column names
df = pd.read_csv('iris.data', delimiter=',', names=col_names)


#checking the first few rows of the DataFrame to check if loaded correctly, note head will default to showing first 5 unless specify more, if you want to see the last 5 you can use tail()
#print(df.head())

#print(df.shape)
#print(df.columns)
#print(df['species'].value_counts())

#Get information about the dataset - number of roms, columns, names of columns, data type, this will allow check if there are any null values
#print(df.info())

#Get a description of the data using the describe() method, as the file contains numerical data, this will display statistics including the count, mean, standard deviation, minimum/maximun value, % Percentiles (note percentiles are used in statistics to give you a number that describes the value that a given percent of the values are lower than. how many of the values are less than the given percentile
#print(df.describe())


#Get a summary of each variable/attribute and output this to a new text file
with open('iris_summary.txt', 'w') as f:
    #Create a variable/heading name for each 
    sepal_length_summary = df['sepal_length'].describe()
    sepal_width_summary = df['sepal_width'].describe()
    petal_length_summary = df['petal_length'].describe()
    petal_width_summary = df['petal_width'].describe()
    
    #Write the summary information to the new text file
    f.write('Summary for Sepal Length:\n{}\n\n'.format(sepal_length_summary))
    f.write('Summary for Sepal Width:\n{}\n\n'.format(sepal_width_summary))
    f.write('Summary for Petal Length:\n{}\n\n'.format(petal_length_summary))
    f.write('Summary for Petal Width:\n{}\n\n'.format(petal_width_summary))

#View the count of each variable in a graph, create a variable for the the dataset to use for graph, ensure to include columns
#iris = pd.read_csv('iris.data', delimiter=',', names=col_names)

#species_counts = df['species'].value_counts()
#species_counts.plot(kind='bar')

#plt.title('Count of each species')
#plt.xlabel('species')
#plt.ylabel('Count')

#plt.show()



#Create a variable for the the dataset, ensure to include columns
#iris = pd.read_csv('iris.data', delimiter=',', names=col_names)

#Histograms for each attribute
#This will add a style to the plot, in this case grey background with white grid lines
#plt.style.use('ggplot')

#plt.figure(figsize = (10, 7))
#x = df["sepal_length"]
#plt.hist(x, bins = 20, color = "purple")
#plt.title("Sepal Length in cm")
#plt.xlabel("Sepal_Length_cm")
#plt.ylabel("Count")
#plt.savefig('Sepal_length.png') 

#plt.figure(figsize = (10, 7))
#x = df.sepal_width
#plt.hist(x, bins = 20, color = "purple")
#plt.title("Sepal Width in cm")
#plt.xlabel("Sepal_Width_cm")
#plt.ylabel("Count")
#plt.savefig('Sepal Width.png')   

#plt.figure(figsize = (10, 7))
#x = df.petal_length
#plt.hist(x, bins = 20, color = "purple")
#plt.title("Petal Length in cm")
#plt.xlabel("Petal_Length_cm")
#plt.ylabel("Count")
#plt.savefig('Petal Length.png') 

#plt.figure(figsize = (10, 7))
#x = df.petal_width
#plt.hist(x, bins = 20, color = "purple")
#plt.title("Petal Width in cm")
#plt.xlabel("Petal_Width_cm")
#plt.ylabel("Count")
#plt.savefig('Petal Width.png') 
#plt.show()


#Create subplots to create histogram of all variables together
#fig, axs = plt.subplots(2, 2, figsize=(10, 10))
#axs[0, 0].hist(df['sepal_length'], bins=10)
#axs[0, 0].set_title('Sepal Length')
#axs[0, 1].hist(df['sepal_width'], bins=10)
#axs[0, 1].set_title('Sepal Width')
#axs[1, 0].hist(df['petal_length'], bins=10)
#axs[1, 0].set_title('Petal Length')
#axs[1, 1].hist(df['petal_width'], bins=10)
#axs[1, 1].set_title('Petal Width')

#Display the subplots
#plt.show()

#Create a pairplot - using seaborn and matplotlib module - this is a useful in visualising the 3 species in pairs and see how they pair up together

#sns.pairplot(df, hue='species')

#plt.show()




#df.plot(kind ="scatter",
#          x ='sepal_length',
 #         y ='petal_length')
#plt.grid()

#plt.show()

#Create a violin plot
#plt.figure(figsize=(14,10))
#plt.subplot(2,2,1)
#sns.violinplot(x='species',y='sepal_length',data=df)
#plt.subplot(2,2,2)
#sns.violinplot(x='species',y='sepal_width',data=df)
#plt.subplot(2,2,3)
#sns.violinplot(x='species',y='petal_length',data=df)
#plt.subplot(2,2,4)
#sns.violinplot(x='species',y='petal_width',data=df)
#plt.show()
