import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('IRIS.csv')
print(df)
df.head()
df.info()
print(df.describe())
print(df['species'].value_counts())
print(df.isnull().sum())
df['sepal_length'].hist()
plt.show()
df['sepal_width'].hist()
plt.show()
df['petal_length'].hist()
plt.show()
df['petal_width'].hist()
plt.show()
colors=['red','green','blue']
species=['Iris-setosa','Iris-versicolor' ,'Iris-virginica' ]
for i in range(3):
    x=df[df['species']==species[i]]
plt.scatter(x['sepal_length'],x['sepal_width'],c=colors[i],label=species[i])
plt.show()
plt.scatter(x['petal_length'],x['petal_width'],c=colors[i],label=species[i])
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.show()
plt.scatter(x['sepal_length'],x['petal_length'],c=colors[i],label=species[i])
plt.xlabel('sepal_length')
plt.ylabel('petal_length')
plt.show()
plt.scatter(x['sepal_width'],x['petal_width'],c=colors[i],label=species[i])
plt.xlabel('sepal_width')
plt.ylabel('petal_width')
plt.show()
#correlation matrix

corr=df.drop('species',axis=1).corr()

sns.heatmap(corr,annot=True,cmap='coolwarm')
plt.show()

#label encoder
le=LabelEncoder()
df['species']=le.fit_transform(df['species'])
print(df['species'])

#model testing
from sklearn.model_selection import train_test_split

x=df.drop('species',axis=1)
y=df['species']
x_test,x_train,y_test,y_train=train_test_split(x,y,test_size=0.2)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
print('Accuracy:',model.score(x_test,y_test))
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)
print('Accuracy:',model.score(x_test,y_test))