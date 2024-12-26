import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('titanic.csv')
print(df)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#Convert the Categorical data to Numerical
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

#handling Null values
df['Age'] =df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'] = df['Embarked'].replace(np.nan,0)

print(df)

#Dropping the unnecessary columns
x = df.drop(columns=['Survived','Name','Ticket','Cabin','PassengerId'], axis=1)
y = df['Survived']

print('XXXX',x)
print('YYYY',y)

#Split the data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)
print('DF',df.shape)
print('X_train',x_train.shape)
print('X_test',x_test.shape)
print('Y_train',y_train.shape)
print('Y_test',y_test.shape)

#Model Initialization
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(x_train, y_train)

#Model Training 
y_pred = NB.predict(x_test)
print('Y_pred',y_pred)
print('Y_test',y_test)

#Model Evaluation 
from sklearn.metrics import accuracy_score
print('Accuracy Score :',accuracy_score(y_test, y_pred)*100,'%')

