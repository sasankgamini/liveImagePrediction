import os
import cv2
import sklearn
import matplotlib.pyplot as plt#
import numpy as np#
from sklearn import preprocessing, neighbors#
import sklearn.model_selection#




datafolder='images'
data=[]
labels=[]
folders=['Stephen Curry', 'Kevin Durant']
for player in folders:
    path=os.path.join(datafolder,player)
    s=os.listdir(path)
    for image in s:
        imgarray=cv2.imread(os.path.join(path,image))
        newarray=cv2.resize(imgarray,(65,65))
##        plt.imshow(newarray)
##        plt.show()
        data.append(newarray)
        labels.append(player)
data=np.array(data)
data=data.reshape(len(data),-1)
print(labels)




data=preprocessing.scale(data)#
data_train,data_test,labels_train,labels_test=sklearn.model_selection.train_test_split(data,labels,test_size=0.1)#
clf=neighbors.KNeighborsClassifier()#
clf.fit(data_train,labels_train)#
accuracy=clf.score(data_test,labels_test)#
print(accuracy)#
try:
    imgarray=cv2.imread('testimage.jpeg')
    cv2.imshow('person',imgarray)
    newarray=cv2.resize(imgarray,(65,65))
except:
    print('unable')
exampledata=np.array(newarray)
exampledata=exampledata.reshape(1,-1)
prediction=clf.predict(exampledata)
print(prediction)
    
