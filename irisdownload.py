#Iris project 
#Author: Sarah Hastings

import pandas as pd

data = pd.read_csv("C:\\Users\\sarah\\Downloads\\iris_csv.csv")

#The function head() will display the top rows of the dataset, the default value of this function is 5, that is it will show top 5 rows when no argument is given to it.

print(data.head())
