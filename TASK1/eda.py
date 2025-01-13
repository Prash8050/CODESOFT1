import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_curve,auc
from sklearn.ensemble import RandomForestClassifier




df=pd.read_csv('TYT.csv')
print(df)
#print(df.info())
#print(df.describe())
print(df.isnull().sum())
sns.countplot(data=df,x='Survived')
plt.show()
sns.countplot(data=df,x='Survived',hue='Sex')
plt.show()
sns.countplot(data=df,x='Survived',hue='Pclass')
plt.show()
sns.distplot(df['Age'].dropna(),kde=False, bins=30)
plt.show()
sns.countplot(data=df,x='SibSp')
plt.show()
d1=df['Age'].fillna(df['Age'].median(),inplace=True)
d2=df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)
print(d1)
print(d2)
print(df.isnull().sum())
df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)
print(df)
df['Sex']=df['Sex'].map({'male':0,'female':1})
df['Embarked']=df['Embarked'].map({'S':0,'C':1,'Q':2})
print(df['Sex'])
print(df['Embarked'])
df['Familysize']=df['SibSp']+df['Parch']+1
print(df['Familysize'])
df.drop(columns=['SibSp','Parch'],inplace=True)
print(df)

x=df.drop('Survived',axis=1)
y=df['Survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train,x_test,y_train,y_test)
model=RandomForestClassifier() 
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,xticklabels=['NO','Yes'],yticklabels=['No','Yes'])
plt.title('confusion matrix')
plt.xlabel('predicted')
plt.ylabel('Acual')
plt.show()
print(classification_report(y_test,y_pred))
feature_importances=pd.DataFrame({'Feature':x.columns,'Importance':model.feature_importances_})
feature_importances.sort_values(by='Importance',ascending=False,inplace=True)
print(feature_importances)