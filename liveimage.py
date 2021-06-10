import numpy as np
import cv2
##import keras
##from keras.models import load_model

##model = load_model('‎⁨savedmodel.h5')

image=cv2.imread('photo1.jpg')
##print(image)
##cv2.imshow('image',image)
##cv2.waitKey()
##cv2.destroyAllWindows()

grayimage=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale to match dataset
##print(grayimage)
##cv2.imshow('gray',grayimage)
##cv2.waitKey()
##cv2.destroyAllWindows()

blurredgray=cv2.GaussianBlur(grayimage,(5,5),0) #we make the iamge blurred, because it gets thick, which is easier to see
##print(blurredgray)
##cv2.imshow('blur',blurredgray)
##cv2.waitKey()
##cv2.destroyAllWindows()

#Threshold the image
ret, im_th = cv2.threshold(blurredgray, 90, 255, cv2.THRESH_BINARY_INV) #makes dark parts black and light parts white and the '_INV' makes the blacks to white and whites to black
##print(im_th)
##cv2.imshow('im_th',im_th)
##cv2.waitKey()
##cv2.destroyAllWindows()

ctrs, hier = cv2.findContours(im_th.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, ctrs, -1,(255,0,0),3)
##cv2.imshow('image',image)
##cv2.waitKey()
##cv2.destroyAllWindows()

rectangles=[]
for eachContour in ctrs:
    rectangles.append(cv2.boundingRect(eachContour))
##print(rectangles)
for eachRectangle in rectangles:
    ROI = im_th[eachRectangle[1]-10:eachRectangle[1]+eachRectangle[3]+10,eachRectangle[0]-10:eachRectangle[0]+eachRectangle[2]+10]
    if ROI.any():
        imgarray=cv2.resize(ROI,(28,28))
        dilatedimg=cv2.dilate(imgarray,(3,3)) #this is to thicken
        dilatedlist=[dilatedimg]
        dilatedarray=np.array(dilatedlist)
        dilatedarray=dilatedarray/255
        print(dilatedarray.shape)
        print('yes')
    cv2.rectangle(im_th,(eachRectangle[0]-10,eachRectangle[1]-10),(eachRectangle[0]+eachRectangle[2]+10,eachRectangle[1]+eachRectangle[3]+10),(255,255,255),2)
cv2.imshow('image',im_th)
cv2.waitKey()
cv2.destroyAllWindows()



