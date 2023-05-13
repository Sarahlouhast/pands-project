# Course: 22-23: HDip in Computing in Data Analytics
## Author: Sarah Hastings
### Subject: Programming and Scripting
#### Lecturer: Andrew Beatty
##### Pandsproject: Fisher’s Iris data set - research, documentation, code, analysis

The project is saved to the following location [pandsproject](https://github.com/Sarahlouhast/pands-project)
Below is a detailed report on the Iris data set, including research, documentation, references used in my research, the code, analysis and what the program will do once executed.

## Background
Firstly, to start off with some background to the Iris flower, its dataset, what is it and why is it so popular. 
Iris is a flowering plant, researchers have measured various features and attributes of the different iris flowers species and recorded these digitally. The iris dataset is a collection of 150 samples of iris flowers, each flower has 4 attributes/features – the sepal length, sepal width, petal length and petal width, all features are in centimetres. Based on the features the species of the flower can be determined, as either Iris-setosa, Iris-versicolor or Iris-virginica. The dataset contains 50 samples of each species. The goal of the iris dataset is to be able to determine or predict the flower species based on its features.  
The Fisher’s paper on the Iris dataset was published by Ronald Fisher in 1936 and became one of the most widely used dataset in data science, used for exploratory data analysis, machine learning, and data visualization and much more. The analysis of the iris dataset is great in many ways, it is a small dataset and easy to work with in this sense but it still provides enough data to produce meaningful results. There is lots of data out there and it is easy to get lost in it all, one major positive is with python for example there is an endless amount of libraries and tools that can be used to help analyse and visualise the data. This analysis leads to a world outside of the standard excel analysis and is the hello world in data science.
Some interesting facts to note along with the Fisher’s paper in 1936, at the same time the Turing machine, a computing device, the first of its kind, was created by Alan Turing in 1936-7. This was the foundation of computer science, Turing also developed Bombe computer to break the Enigma Code during World War II, which was estimated to have shortened war by more than 2 years and saved 14 million lives. Also 1936 was significant in that Margaret Hamilton was born, she went on to become a computer scientist, one of the first computer software programmers, she created the term software engineer and led the team that created the onboard flight software for the Apollo missions, Apollo 11, landing astronauts on the moon. A significant year over all 1936, with Fisher, Turing and Hamilton each making massive, significant contributions to their respective fields which still have an impact and are used to this current day.   

## Detail on the program [analysis.py](https://github.com/Sarahlouhast/pands-project/blob/main/analysis.py) 

Firstly, to work with the dataset I will need to import libraries, I choose to use the below libraries for this project. 
###### import pandas as pd
###### import numpy as np
###### import matplotlib.pyplot as plt
###### import seaborn as sns
###### from sklearn.model_selection import train_test_split
###### from sklearn.preprocessing import StandardScaler
###### from sklearn.metrics import accuracy_score,confusion_matrix
###### from pandas.plotting import andrews_curves

It is necessary to import pandas to allow for data analysis and manipulation. Import numpy to work with numerical data, arrays. Import matplotlib for use in creating static, animated, and interactive visualizations in python, pyplot is imported as a submodule of the matplotlib library and is used to visualize diagrams/visual data/plots. Seaborn is similar to matplotlib but gives a variety of visualization patterns which can be used in the analysis and is useful for plotting visual relationship between data. Importing andews curves is another method which is useful in visualising data. 
To perform analysis on the data it is necessary to import tools from other libraries such as sklearn, this contains tools that will allow you to train and test the data, classification and regression tools and much more, all which will help in predicting the class/species of the iris flower based on its specific features. 
To work with the dataset it is necessary to import the data and load it into a dataframe. A dataframe is a 2 dimensional data structure, like a 2 dimensional array, or a table with rows and columns and can be done through pandas. I downloaded the data from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris) and saved it to my repository. The data file is CSV file (comma separated files) and can be read in using pd.read_csv() function. 
The downloaded file does not contain the necessary column names, these are added using the below and then read into the dataframe to include the column names. 
```
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv('iris.data', delimiter=',', names=col_names)
```
Before performing any analysis, testing or modelling it is necessary to check the data file, ensure it loaded correctly and does not contain any bad data/empty cells/duplicates/incorrect formatting. 
To do this you can check the first few rows of the dataframe to check if it loaded correctly, using the head() function, note this will default to showing the first 5 rows unless you specify more, if you want to see the last 5 you can use the tail() function. The output of the print(df.head()) will display the below.
```
   sepal_length  sepal_width  petal_length  petal_width      species
0           5.1          3.5           1.4          0.2  Iris-setosa
1           4.9          3.0           1.4          0.2  Iris-setosa
2           4.7          3.2           1.3          0.2  Iris-setosa
3           4.6          3.1           1.5          0.2  Iris-setosa
4           5.0          3.6           1.4          0.2  Iris-setosa
```
It can be useful to view the shape of the file the shape of the file using the print(df.shape) function which display the following, tell you the number of rows and columns. 
```
(150, 5)
```

Using the print(df.columns) will give you information on the columns, including column name. This will output the below.
```
Index(['sepal_length', 'sepal_width', 'petal_length', 'petal_width',
       'species'],
      dtype='object')
```
The count() function can be used to count the number of cells, using the print(df['species'].value_counts()) will output the below, giving information on the species, the names of calls, the count of each and the data type. 
```
species
Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50
Name: count, dtype: int64
```

To get more detailed information about the dataset in one function, such as the number of rows, columns, names of columns, data type, use can use the info() function, this will allow check if there are any null values. Using this function will print the below output.

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
 #   Column        Non-Null Count  Dtype
---  ------        --------------  -----
 0   sepal_length  150 non-null    float64
 1   sepal_width   150 non-null    float64
 2   petal_length  150 non-null    float64
 3   petal_width   150 non-null    float64
 4   species       150 non-null    object
dtypes: float64(4), object(1)
memory usage: 6.0+ KB
None 
```

It is worth noting that while the info() function checks for null values, there is also another function, .isnull(), which can be used on its own for cleaning and detecting missing values, to find missing values i.e NaN, which can occur due to several reasons. If data is missing, it will display True else False.

To get a more detailed description of the data you can use the describe() function. As the file contains numerical data, this function will display statistics information including the count, mean, standard deviation, minimum/maximum value, % Percentiles (note percentiles are used in statistics to give you a number that describes the value that a given percent of the values are lower than, how many of the values are less than the given percentile). The print(df.describe()) will output the below.
```
       sepal_length  sepal_width  petal_length  petal_width
count    150.000000   150.000000    150.000000   150.000000
mean       5.843333     3.054000      3.758667     1.198667
std        0.828066     0.433594      1.764420     0.763161
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000
```

If you don’t want to use the describe function and want specific information or want to find this seperately, you can do so, for example, if you want to find the minimum and maximum from a column, in this case I will use the sepal length. I will create a variable for each min, max and read in the specific column with the function. 
```
min_data=df["sepal_length"].min()
max_data=df["sepal_length"].max()  
print("Minimum:",min_data, "\nMaximum:", max_data)
```
This will output the below.
```
Minimum: 4.3 
Maximum: 7.9
```

It can be useful also, if you are more of a visual person to view the count of each variable in a graph. The below code will display this, firstly creating a variable for the species count which contains the count function to count the specific column of species, this will be displayed in a bar plot, with a title and labels added for each axis and the show() function to display this. 

```
species_counts = df['species'].value_counts()
species_counts.plot(kind='bar')
plt.title('Count of each species')
plt.xlabel('species')
plt.ylabel('Count')
plt.show()
```

Next it can be useful to get a summary of each feature of the iris species and output this to a text file. In this case I will call the file iris_summary.txt. I will create a variable/heading name for each attribute, sepal_length, sepal_width, petal_length, petal_width. Each variable which will include the describe() function in order to get the summary of that attribute . I will then write the summary information to the new text file using write() method which writes a specified text to the file. The below code will execute this. 

```
with open('iris_summary.txt', 'w') as f:
    sepal_length_summary = df['sepal_length'].describe()
    sepal_width_summary = df['sepal_width'].describe()
    petal_length_summary = df['petal_length'].describe()
    petal_width_summary = df['petal_width'].describe()
    
    f.write('Summary for Sepal Length:\n{}\n\n'.format(sepal_length_summary))
    f.write('Summary for Sepal Width:\n{}\n\n'.format(sepal_width_summary))
    f.write('Summary for Petal Length:\n{}\n\n'.format(petal_length_summary))
    f.write('Summary for Petal Width:\n{}\n\n'.format(petal_width_summary))
```

Next I will do some plotting and visualizing of the data to display and help in the understanding of the relationship between the features and show patterns, correlations and trends in the data.
I will create a histogram for each attribute, using the dataframe ‘df’ containing the file data. Histograms allow the inspection of data displaying the frequency distribution of the variables. 
I will create a variable for each attribute, add a style to the plot using plt.style.use('ggplot'), in this case a grey background with white grid lines will be added to the display. And I will save the histogram plt.savefig('xxxx.png'). Just to note here if you want to save your plot use can do so but ensure this function is called before the show function, otherwise if it is after the show function your image will be blank. I will also create subplots to display a histogram of all the variables together also, to view these in one shot can be useful too. 
The below code is an example of the histogram for the sepal_length attribute.

```
plt.style.use('ggplot')
plt.figure(figsize = (10, 7))
x = df["sepal_length"]
plt.hist(x, bins = 20, color = "purple")
plt.title("Sepal Length in cm")
plt.xlabel("Sepal_Length_cm")
plt.ylabel("Count")
plt.savefig('Sepal_length.png')
plt.show()
```

From the histograms of each variable, the values with longer plots highlight that more values are concentrated there, you can see how each variable is distributed across the dataset, showing the frequency, count, giving you an understanding of the pattern or shape of the distribution across the dataset. 

Next I will create a scatter plot of variables for petals, sepals to show the relationship between the two variables. And save a copy of the output to a png file, from these plots it can be seen that the iris setosa is more separated from the other species.

Following this I will create a pairplot using the seaborn and the matplotlib module, this is extremely useful in visualising the 3 species in pairs, how they pair up together, representing the positive and negative correlation between the variables, all at once, in a quick view shot, instead of doing this individually. I found this much more beneficial than the scatterplot, mainly because it displays a full matrix of the relationships between each variable. For this plot the species will be the variable to map into the differentiating colours. Each of these species is compared with each of the other variables in the dataset. Like the scatterplot result, one clear observation which can be taken from the pairplot is that the Iris-Setosa (in blue) is separated from both other species in all the features, while separating versicolor from virginica is much harder as they have some overlap. 

Another usual tool for visualization is a violin plot, this will be a plot of all the input variables (the features) against the output variable which is the species. The violin plot will show density of the length and width of the species. The thinner part highlights that there is less density whereas the wider part conveys higher density, a take-away can be that the Iris-Setosa class has a smaller petal length and petal width compared to the other classes. These can be viewed from the png files saved to the repository.

Staying with the visualisation tools another technique from the pandas library is called Andrews Curves.
This will show the relationship between the iris flower features and the species, plotting the curves for each iris flower grouped by species. Where the curves are distinct and separate from the other curves, you can see in the case of the Iris-setosa species this has different feature values compared to Iris-versicolor and Iris-virginica. Whereas where the curves overlap, you can see that the feature values are similar across the species at these stages.

Next up I will analyse the dataset in more detail, with a correlation matrix, exploring the correlations between the different variables, analyzing the strength and direction of the relationship between pairs of variables. And I will display this in a heatmap for visualization.
Each cell in the table will shows the correlation between the two variables. The value in the matrix will range from -1 to 1, where -1 indicates a negative low correlation, 0 indicates no correlation, and 1 indicates a positive high correlation. It is worth noting that I will create a numeric variable of the dataframe for the matrix to take in the numeric data only. The heatmap will be styled using seaborn tools such as colourmap and a copy of this will be saved to my repository.
The output of the matrix table will display the below. 
```
              sepal_length  sepal_width  petal_length  petal_width
sepal_length      1.000000    -0.109369      0.871754     0.817954
sepal_width      -0.109369     1.000000     -0.420516    -0.356544
petal_length      0.871754    -0.420516      1.000000     0.962757
petal_width       0.817954    -0.356544      0.962757     1.000000
```
From this table and the heatmap visual, you can see where two variables are close to +1 with a positive correlation, it shows a good relationship, when one variable increases, the other variable is likely to increase as well, if you look at the example petal_length and petal_width (value 0.96), which means that as the petal length increases, the petal width also tends to increase. 
In the case where the value is close to 0, it means that there is little to no relationship between them, take a look at sepal_length and petal_width have a weak correlation coefficient (0.82), you can take away there may not be a strong relationship between these two variables.
To summarize the correlation matrix is extremely useful in displaying how the different variables are related to each other, which can be useful in the next stage for understanding the data, making decisions about testing, modelling, accuracy and predictability.

In the final stage, I will look at testing, modelling, accuracy and predictability of the species. I will start with the train and test function to train the model and test the dataset. This is a method to measure the accuracy of the model, which can then be used to predict the species based on the features. 
Firstly, I will need to split the data into train and test sets, I train the model using the training set, I test the model using the testing set. Train the model means create the model. Test the model means test the accuracy of the model. The default value for the train_test_split splits the data into 75% training data and 25% test data.  To get the value of the train and test data you can use the shape method. I will create a variable to store the train, test data. Using the iloc to select the specific rows/columns from the dataframe.
Once this step is complete it helps in working towards predicting the species based on the features and the accuracy of this. To do so I will classify the species using logistic regression and a confusion matrix. Logistic regression has been imported already from the sklearn library. A model of logistic regression variable is created and I will pass the earlier training datasets through this model and then predict the results with the predict method. This will contain the species names in the form of an array. To find the accuracy of the model I will use the accuracy method and this will output the below.
```
Accuracy of the model is 97.37 %.
```
This will tell you how accurately the model built will predict the species values. 
Using Logistic Regression, allows you to classify the iris flower samples into their respective species, from this model you can see an accuracy score of 97.37%, which shows that the model built is very accurate in predicting the species. Although it is important to note that this is a sample output, and accuracy may vary depending on the data/splits/sample sizes.

In this case it is good to have another method to use in predicting species, here I will use a confusion matrix for this. 
The confusion matrix will display a matrix with actual values and predicted values based on the data from the train and test split model created earlier. This will give a breakdown of the predictions made by the model. The output of the matrix resulted in the below.
```
[[13  0  0]
 [ 0 15  1]
 [ 0  0  9]]
``` 
The matrix shows the number of true positives, true negatives, false positives, and false negatives. I will display this in a heatmap, where the different colours represent the intensity or magnitude of the values in the matrix, which will help in visualizing patterns and distributions of the predictions.
The darker or more intense colours will represent higher values indicating accurate predictions, while lighter or less intense will represent lower values, indicating incorrect predictions.
By using the confusion matrix along with logistic regression both give insights into the classification of the model built and provide accuracy on the predictions made. 

The below code will execute these functions.

```
x = df.iloc[:,:-1].values
y = df.iloc[:,4].values
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)
cm=(confusion_matrix(y_test,y_pred))
print(cm)
display confusion matrix
sns.heatmap(cm, annot=True, cmap='YlGnBu', fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('Actual Labels')
plt.show()
accuracy=accuracy_score(y_test,y_pred)*100
print("Accuracy of the model is {:.2f}".format(accuracy),"%.")
```
## Conclusion 
Overall it is evident that there is a massive world of data and exploring out there. There is endless analysis that can be done. This project is a summary of the iris dataset, a background to it, the possible options that can be used in investigating the dataset, providing insights into the analysis, the review of the data file itself, options on how to train and test data to allow you to make predictions on the species based on its specific features using various methods such as classifications, logistic regression and confusion matrix. 

## References
* <https://www.w3schools.com/python/pandas/default.asp>
* <https://www.datacamp.com/cheat-sheet/pandas-cheat-sheet-for-data-science-in-python>
* <https://res.cloudinary.com/dyd911kmh/image/upload/v1676302204/Marketing/Blog/Pandas_Cheat_Sheet.pdf>
* <https://towardsdatascience.com/eda-of-the-iris-dataset-190f6dfd946d>
* <https://www.analyticsvidhya.com/blog/2021/05/shape-of-data-skewness-and-kurtosis/#:~:text=The%20skewness%20is%20a%20measure,pushed%20towards%20the%20left%20side).>
* <https://www.codespeedy.com/plotting-violin-plots-in-python-using-the-seaborn-library/>
* <https://www.geeksforgeeks.org/violin-plot-for-data-analysis/>
* <https://www.geeksforgeeks.org/make-a-violin-plot-in-python-using-matplotlib/>
* <https://www.w3schools.com/python/python_ml_percentile.asp>
* <https://matplotlib.org/stable/tutorials/colors/colormaps.html>
* <https://www.analyticsvidhya.com/blog/2022/06/iris-flowers-classification-using-machine-learning/>
* <https://www.w3schools.com/python/python_ml_confusion_matrix.asp#:~:text=What%20is%20a%20confusion%20matrix,the%20predictions%20we%20have%20made.>
* <https://vitalflux.com/python-creating-scatter-plot-with-iris-dataset/>
* <https://www.theguardian.com/technology/2019/jul/13/margaret-hamilton-computer-scientist-interview-software-apollo-missions-1969-moon-landing-nasa-women>
* <https://dillinger.io/>
* <https://www.kaggle.com/code/rahulrajpandey31/logistic-regression-from-scratch-iris-data-set/notebook>
* <https://www.kaggle.com/code/ekapylski/iris-dataset-visualization>
* <https://learn.theprogrammingfoundation.org/getting_started/intro_data_science/module4/>
* <https://www.w3schools.com/python/python_ml_train_test.asp>
* <https://www.britannica.com/biography/Alan-Turing>