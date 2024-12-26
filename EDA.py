import numpy as np
import pandas as pd
df = pd.read_csv('titanic.csv')

# pd.set_option('display.max_columns', None) #to display all the columns in the dataframe
# print(df)

print(df.info()) #to get the information about the dataframe
print(df.head(10)) #to get the first 10 rows of the dataframe
print(df.tail(10)) #to get the last 10 rows of the dataframe
print(df.columns) #to get the number of columns and the name of the columns in the dataframe
print(df.shape) #to get the number of rows and columns in the dataframe
print(df.duplicated().sum()) #to check for duplicate values in the dataframe and give the number of duplicate values
print(df.isnull().sum()) #to check for null values in the dataframe and give the number of null values

import matplotlib.pyplot as plt
import missingno as ms
ms.bar(df,figsize=(10,5),color='green')
plt.title('Shows the missing data in the dataframe', size=15 , color='green')
plt.show()

df.drop(['Cabin'], axis=1, inplace=True) #to drop the Cabin column and inplace=True to update the dataframe
# df.dropna(inplace=True) #to drop the rows with null values and inplace=True to update the dataframe
print(df['Embarked'].value_counts()) #to get the count of the values in the Embarked column
print(df['Embarked'].unique()) #to get the unique values in the Embarked column
print(df['Embarked'].nunique()) #to get the number of unique values in the Embarked column
print(df['Age'].mean()) #to get the mean of the Age column
print(df['Age'].median()) #to get the median of the Age column
print(df['Age'].std()) #to get the standard deviation of the Age column
print(df['Age'].min()) #to get the minimum value of the Age column
print(df['Age'].max()) #to get the maximum value of the Age column
# print (df['Age'].sort_values(ascending=False)) #to sort the values in the Age column in descending order if ascending is set to True tehn the numbers will sorted in the ascending order
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].median()) #to fill the null values in the Embarked column with the median of the Embarked column
print(df['Embarked'].value_counts()) #to get the count of the values in the Embarked column
df['Age'] = df['Age'].fillna(df['Age'].mean()) #to fill the null values in the Age column with the mean of the Age column
print(df.describe()) #to get the summary statistics of the dataframe(mean, median, standard deviation, minimum, maximum, etc.)
