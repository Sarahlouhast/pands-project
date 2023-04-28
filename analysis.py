#Author: Sarah Hastings
#Course: 22-23: HDip in Computing in Data Analytics
#Module: Programming and Scripting
#Lecturer: Andrew Beatty
#Pandsproject: Fisher’s Iris data set - research, documentation, code, analysis


#Firstly import pandas to allow for data analysis, manipulation
import pandas as pd
#Use Pyplot, a submodule of the Matplotlib library to visualize the diagram/use for visual data/plots etc 
import matplotlib.pyplot as plt
#read iris.data file into a DataFrame, allowed through use of pandas imported
df = pd.read_csv('iris.data', delimiter=',')

#note the iris.data file does not contain the necessary column names, these are added using the below
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

#read iris.data file into a DataFrame with column names
df = pd.read_csv('iris.data', delimiter=',', names=col_names)


#checking the first few rows of the DataFrame to check if loaded correctly, note head will default to showing first 5 unless specify more, if you want to see the last 5 you can use tail()
#print(df.head())

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


#iris = pd.read_csv('iris.data', delimiter=',', names=col_names)

#species_counts = iris['species'].value_counts()
#species_counts.plot(kind='bar')

#plt.title('Count of each species')
#plt.xlabel('species')
#plt.ylabel('Count')

#plt.show()



#Create a variable for the the dataset, ensure to include columns
iris = pd.read_csv('iris.data', delimiter=',', names=col_names)

#Create subplots to create histogram of all variables together - can do seperate also - 
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

#Create a histogram for each variable
axs[0, 0].hist(iris['sepal_length'], bins=10)
axs[0, 0].set_title('Sepal Length')
axs[0, 1].hist(iris['sepal_width'], bins=10)
axs[0, 1].set_title('Sepal Width')
axs[1, 0].hist(iris['petal_length'], bins=10)
axs[1, 0].set_title('Petal Length')
axs[1, 1].hist(iris['petal_width'], bins=10)
axs[1, 1].set_title('Petal Width')

# Display the subplots
plt.show()