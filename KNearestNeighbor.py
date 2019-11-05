import operator
from collections import Counter
import numpy as np

class KNearestNeighbors:
    def __init__(self,k):
        self.k=k

    def fit(self,x_train,y_train):
        self.x_train=x_train
        self.y_train=y_train
        print("Training Done")

    def predict(self,x_test):
        distance={}
        counter=0
        a=self.x_train.shape[1]
        classification=np.array([],dtype='int')
        for i in x_test:
            b=0
            for j in self.x_train:
                b=b+np.sum((i-j) ** 2)
                b=b**1/2
                distance[counter]=b
                b=0
                counter=counter+1
            distance=sorted(distance.items(),key=operator.itemgetter(1))
            c=self.classify(distance[0:self.k])
            classification=np.append(classification,c)
            counter=0
            distance={}
        return classification
    def classify(self,distance):
        label=[]
        for i in distance:
            label.append(self.y_train[i[0]])

        return Counter(label).most_common()[0][0]