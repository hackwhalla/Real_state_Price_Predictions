import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,ConfusionMatrixDisplay,classification_report
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

x  =data.iloc[:,:-1]
y = data.iloc[:,-1]

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

m6 = KNeighborsRegressor()
m6 = m6.fit(x_train,y_train)

y6 = m6.predict(x_test)

y_pred = y6
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

#print(f"\n📊 {name} Results:")
print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.4f}")
print(f"MSE      : {mse:.4f}")
print(f"RMSE     : {rmse:.4f}")