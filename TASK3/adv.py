import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

df=pd.read_csv('advertise.csv')
print(df)
df.head()
df.info()
df.describe()
print(df.isnull().sum())
print(df.duplicated().sum())
df.drop(columns=['Unnamed: 0'],inplace =True)
print(df)
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Drop duplicates if any
df = df.drop_duplicates()
print(df)
sns.pairplot(df, x_vars=['TV Ad Budget ($)', 'Radio Ad Budget ($)', 'Newspaper Ad Budget ($)'], y_vars='Sales ($)', kind='scatter')
plt.show()

#c0rr matrix
corr=df.corr()
sns.heatmap(corr,annot=True,cmap='coolwarm')
plt.title('correlation matrix')
plt.show()

#hitogram for numeric feautures
df.hist(bins=30,figsize=(10,10))
plt.suptitle('histogram for features',y=1.02)
plt.show()

#feature selection and spliting
x=['TV Ad Budget ($)','Radio Ad Budget ($)','Newspaper Ad Budget ($)']
y='Sales ($)'


X = df[['TV Ad Budget ($)', 'Radio Ad Budget ($)', 'Newspaper Ad Budget ($)']]  # Independent variables
y = df['Sales ($)']  # Dependent variable

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

#model buidling
model=LinearRegression()
model.fit(X_train,y_train)
#predictions
y_pred=model.predict(X_test)

#evaluate model
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(mse)
print(r2)
plt.scatter(y_test,y_pred,color='blue')
plt.plot([y_test.min(), y_test.max()],[y_test.min(),y_test.max()], color='red', linewidth=2)
plt.title('Actual vs Predicted Sales')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.grid()
plt.show()

#coeficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("Feature Coefficients:")
print(coefficients)
