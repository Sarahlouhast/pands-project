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
#useful for plotting visual relationship between data 
from pandas.plotting import andrews_curves
#split the dataset into train and test
from sklearn.model_selection import train_test_split
#training model
from sklearn.preprocessing import StandardScaler
#logical regression
from sklearn.linear_model import LogisticRegression
#view accuracy/predictability
from sklearn.metrics import accuracy_score,confusion_matrix
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

#data.isnull() Cleaning and detecting missing values , try to find the missing values i.e NaN, which can occur due to several reasons.
#if there is data is missing, it will display True else False.
#print(df.isnull())


#Get a description of the data using the describe() method, as the file contains numerical data, this will display statistics including the count, mean, standard deviation, minimum/maximum value, % Percentiles (note percentiles are used in statistics to give you a number that describes the value that a given percent of the values are lower than. how many of the values are less than the given percentile
#print(df.describe())

#if dont want to use describe and manual find values for eg min/max - Extracting minimum and maximum from a column. Identifying minimum and maximum integer, from a particular column or row can also be done in a dataset.
#min_data=df["sepal_length"].min()
#max_data=df["sepal_length"].max()
  
#print("Minimum:",min_data, "\nMaximum:", max_data)

#View the count of each variable in a graph, create a variable for the the dataset to use for graph, ensure to include columns

#species_counts = df['species'].value_counts()
#species_counts.plot(kind='bar')

#plt.title('Count of each species')
#plt.xlabel('species')
#plt.ylabel('Count')

#plt.show()




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
#plt.style.use('ggplot')
#fig, axs = plt.subplots(2, 2, figsize=(10, 7))
#axs[0, 0].hist(df['sepal_length'], bins=10, color = "purple")
#axs[0, 0].set_title('Sepal Length')
#axs[0, 1].hist(df['sepal_width'], bins=10, color = "purple")
#axs[0, 1].set_title('Sepal Width')
#axs[1, 0].hist(df['petal_length'], bins=10, color = "purple")
#axs[1, 0].set_title('Petal Length')
#axs[1, 1].hist(df['petal_width'], bins=10, color = "purple")
#axs[1, 1].set_title('Petal Width')


#plt.savefig('Histogram of all variables') 
#Display the subplots
#plt.show()


#Scatter plot of variables for petals, sepals
#setosa = df[df.species == "Iris-setosa"]
#versicolor = df[df.species=='Iris-versicolor']
#virginica = df[df.species=='Iris-virginica']

#fig, ax = plt.subplots()
#fig.set_size_inches(13, 7) # adjusting the length and width of plot

# lables and scatter points
#ax.scatter(setosa['petal_length'], setosa['petal_width'], label="Setosa", facecolor="blue")
#ax.scatter(versicolor['petal_length'], versicolor['petal_width'], label="Versicolor", facecolor="green")
#ax.scatter(virginica['petal_length'], virginica['petal_width'], label="Virginica", facecolor="red")


#ax.set_xlabel('petal_length')
#ax.set_ylabel('petal_width')
#ax.grid()
#ax.set_title('Iris petals')
#ax.legend()
#plt.savefig('Iris petals')

#sepal
#fig, ax = plt.subplots()
#fig.set_size_inches(13, 7) # adjusting the length and width of plot
# lables and scatter points
#ax.scatter(setosa['sepal_length'], setosa['sepal_width'], label="Setosa", facecolor="blue")
#ax.scatter(versicolor['sepal_length'], versicolor['sepal_width'], label="Versicolor", facecolor="green")
#ax.scatter(virginica['sepal_length'], virginica['sepal_width'], label="Virginica", facecolor="red")

#ax.set_xlabel('sepal_length')
#ax.set_ylabel('sepal_width')
#ax.grid()
#ax.set_title('Iris sepals')
#ax.legend()
#plt.savefig('Iris sepals')

#plt.show()




#Create a pairplot - using seaborn and matplotlib module - this is a useful in visualising the 3 species in pairs and see how they pair up together

#sns.pairplot(df, hue='species')
#plt.savefig('Pairplot Iris Dataset.png') 
#plt.show()


#Create a violin plot
#plt.figure(figsize=(14,10))  # adjusting the length and width of plot
#plt.subplot(2,2,1)
#sns.violinplot(x='species',y='sepal_length',data=df)
#plt.subplot(2,2,2)
#sns.violinplot(x='species',y='sepal_width',data=df)
#plt.subplot(2,2,3)
#sns.violinplot(x='species',y='petal_length',data=df)
#plt.subplot(2,2,4)
#sns.violinplot(x='species',y='petal_width',data=df)
#plt.savefig('Violin plot Iris Dataset')
#plt.show()

#Andrews Curves 
#Create the Andrews curves plot with colormap
#andrews_curves(df, 'species', colormap='plasma')
#plt.title('Andrews Curves Plot - Iris Dataset')
#plt.savefig('Andrews Curves Iris Dataset')
#plt.show()



#Correlation matrix, heatmap
#numeric_columns = df.select_dtypes(include='number')
#correlation_matrix = numeric_columns.corr()
#print(correlation_matrix)
#sns.heatmap(correlation_matrix, cmap = "YlGnBu", linecolor = 'white', linewidths = 1, annot = True)
#plt.title('Correlation Heatmap - Iris Dataset')
#plt.savefig('Correlation Heatmap - Iris Dataset')
#plt.show()


#Split the Data Into Train and Test Datasets
#x = df.iloc[:,:-1].values
#y = df.iloc[:,4].values
#x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
#print(x_train.shape)
#print(x_test.shape)
#print(y_train.shape)
#print(y_test.shape)

#Create the Model (Classification - logistic regression)
#model=LogisticRegression()
#model.fit(x_train,y_train)
#y_pred=model.predict(x_test)
#print(y_pred)
#cm=(confusion_matrix(y_test,y_pred))
#print(cm)
#display confusion matrix
#sns.heatmap(cm, annot=True, cmap='YlGnBu', fmt='d')
#plt.title('Confusion Matrix')
#plt.xlabel('Predicted Labels')
#plt.ylabel('Actual Labels')
#plt.show()


#accuracy=accuracy_score(y_test,y_pred)*100
#print("Accuracy of the model is {:.2f}".format(accuracy),"%.")
