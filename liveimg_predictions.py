import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
import cv2
from keras.models import load_model

##((trainimgs, trainlabels),(testimgs,testlabels))= mnist.load_data()
##class_names=[0,1,2,3,4,5,6,7,8,9]
##trainimgs=trainimgs/255
##testimgs=testimgs/255
##    
###building the model(blueprint)
##model=keras.Sequential([
##    keras.layers.Flatten(input_shape=(28,28)),
##    keras.layers.Dense(512,activation='relu'), #activation if it passes certian threshold
##    keras.layers.Dense(10,activation='softmax') #gives percentages for each number in third layer
##    ])
##
###Compile the model/properties of model(giving extra features/personalizing)
##model.compile(optimizer='adam',
##              loss='sparse_categorical_crossentropy',
##              metrics= ['accuracy'])
##
###Train the model
##model.fit(trainimgs,trainlabels,epochs=5)
##
###saves a model so it will be faster and won't have to train each time you run it
##model.save('savedmodel.h5')

###Test the model
##test_loss, test_acc = model.evaluate(testimgs, testlabels)
##print(test_acc)  #accuracy of test

###Predictions
##predictions=model.predict(testimgs)
##print(predictions[1]) #predictions of first test image
##print(np.argmax(predictions[1])) #index of highest prediction
##
##print(class_names[np.argmax(predictions[1])])
##
##plt.imshow(testimgs[1])
##plt.show()



##model = load_model('savedmodel05.h5')

image=cv2.imread('photo1.jpg')
print(image)
##cv2.imshow('image',image)
##cv2.waitKey()
##cv2.destroyAllWindows()

grayimage=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale to match dataset
print(grayimage)
##cv2.imshow('gray',grayimage)
##cv2.waitKey()
##cv2.destroyAllWindows()
##
##blurredgray=cv2.GaussianBlur(grayimage,(5,5),0) #we make the iamge blurred, because it gets thick, which is easier to see
####print(blurredgray)
####cv2.imshow('blur',blurredgray)
####cv2.waitKey()
####cv2.destroyAllWindows()
##
###Threshold the image
##ret, im_th = cv2.threshold(blurredgray, 90, 255, cv2.THRESH_BINARY_INV) #makes dark parts black and light parts white and the '_INV' makes the blacks to white and whites to black
####print(im_th)
####cv2.imshow('im_th',im_th)
####cv2.waitKey()
####cv2.destroyAllWindows()
##
##ctrs, hier = cv2.findContours(im_th.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
##cv2.drawContours(image, ctrs, -1,(255,0,0),3)
####cv2.imshow('image',image)
####cv2.waitKey()
####cv2.destroyAllWindows()
##
##rectangles=[]
##for eachContour in ctrs:
##    rectangles.append(cv2.boundingRect(eachContour))
####print(rectangles)
##for eachRectangle in rectangles:
##    ROI = im_th[eachRectangle[1]-20:eachRectangle[1]+eachRectangle[3]+20,eachRectangle[0]-20:eachRectangle[0]+eachRectangle[2]+20]
##    if ROI.any():
##        imgarray=cv2.resize(ROI,(28,28))
##        dilatedimg=cv2.dilate(imgarray,(3,3)) #this is to thicken
##        dilatedlist=[dilatedimg]
##        dilatedarray=np.array(dilatedlist)
##        dilatedarray=dilatedarray/255
####        print(dilatedarray.shape)
####        print('yes')
##        predictions=model.predict(dilatedarray)
##        print(predictions[0])
##        print(np.argmax(predictions[0]))
##        cv2.putText(im_th, str(np.argmax(predictions[0])), (eachRectangle[0]-10, eachRectangle[1]-50),cv2.FONT_HERSHEY_DUPLEX, 2, (255,255,255), 2)
##    cv2.rectangle(im_th,(eachRectangle[0]-10,eachRectangle[1]-10),(eachRectangle[0]+eachRectangle[2]+10,eachRectangle[1]+eachRectangle[3]+10),(255,255,255),2)
##cv2.imshow('image',im_th)
##cv2.waitKey()
##cv2.destroyAllWindows()
##
##
###HW: put predicted images on top of original rectangles on image
###use cv2.putText
