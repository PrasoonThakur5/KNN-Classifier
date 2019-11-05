import numpy as np
import pandas as pd
from KNearestNeighbor import KNearestNeighbors

data = pd.read_csv("Social_Network_ads.csv")

x=data.iloc[:,2:4].values
y=data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,ytest=train_test_split(x,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

knn=KNearestNeighbors(k=50)

knn.fit(x_train,y_train)

def predict_new():
    age=int(input("Enter The Age - "))
    salary = int(input("Enter The Salary - "))
    x_new=np.array([[age],[salary]]).reshape(1,2)

    x_new=scaler.transform(x_new)
    result=knn.predict(x_new)

    if result==0:
        print("Will Not Purchase")
    else:
        print("Will Purchase")

predict_new()