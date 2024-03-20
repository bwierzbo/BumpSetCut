import os
import cv2 as cv
import numpy as np
import keras
import tensorflow as tf
from keras.models import load_model

# Load your trained model
model = load_model('../BumpSetCut/my_model.keras')


#this is the path of where getball outputs its images into
imageoutpath = ('../BumpSetCut/images')


#This is where you set the path to the volleyball video
videoCapture = cv.VideoCapture('../BumpSetCut/video/longerClip.mp4/')
videoCapture.set(cv.CAP_PROP_BUFFERSIZE, 2)
prevCircle = None
dist = lambda x1,y1,x2,y2: (x1-x2)**2+(y1-y2)**2

backSub = cv.createBackgroundSubtractorKNN()
n=0





def preprocess_for_model(img, size=(224, 224)):
    img = cv.resize(img, size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Model expects batch dimension
    return img



while True:
    ret, frame = videoCapture.read()
    if not ret: break
    
    frame = cv.resize(frame, (1920,1080))

    mask = backSub.apply(frame)


    mask = cv.GaussianBlur(mask, (13, 13),0)
    ret,mask = cv.threshold(mask,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)


    #See for HoughCircles perameter description https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
    circles = cv.HoughCircles(mask, cv.HOUGH_GRADIENT_ALT, 1.5, 1, param1=250, param2= 0.8, minRadius=3, maxRadius=40)

    if circles is not None:
            circles = np.uint16(np.around(circles))
            chosen = None
            for i in circles[0, :]:
                if chosen is None: chosen = i
                if prevCircle is not None:
                    if dist(chosen[0],chosen[1],prevCircle[0],prevCircle[1]) <= dist(i[0],i[1],prevCircle[0],prevCircle[1]):
                        chosen = i
            

            #chosen[2] is radius 
            #Using the radius to get the starting x,y coordinates and the width and height for the image cut
            rx = chosen[0] - chosen[2]
            ry = chosen[1] + chosen[2]
            rw = chosen[2]*2
            rh = chosen[2]*2
            
            #rectangle shows what it is cutting for the classifier
            #cv.rectangle(frame,((rx-5), ry+5), ((rx+rw+5),(ry-rh-5)),(255,0,255), 3)
            #print(chosen[2])


            #cutting images from black and white mask
            cut_m = mask[ry - rh - 5 : ry + 5, rx - 5 : rx + rw + 5]

            #cutting images from color mask
            cut_f = frame[ry - rh - 5 : ry + 5, rx - 5 : rx + rw + 5]

            #Splicing the images to cutout the background but keep the circle detected in
            cut_c = cv.bitwise_and(cut_f,cut_f,mask = cut_m)

            

            #Sometimes the frames return a null value so this checks it and will not write null image as it will crash the program
            if cut_c is None:
                 print("none")
            else:
                # Preprocess the cutout for the model
                preprocessed_img = preprocess_for_model(cut_c)

                # Use the model to predict
                prediction = model.predict(preprocessed_img)
                predicted_class = 'notBall' if prediction[0][0] > 0.5 else 'ball'
                print(f"Detected: {predicted_class} with confidence {prediction[0][0]}")

                #These draw the circles being detected in each frame 
                if predicted_class == 'ball':
                    cv.circle(frame, (chosen[0], chosen[1]), chosen[2], (0,255,0), 3)

                elif predicted_class == 'notBall':
                    #cv.circle(frame, (chosen[0], chosen[1]), 1, (255,0,0), 3)
                    cv.circle(frame, (chosen[0], chosen[1]), chosen[2], (0,0,255), 3)
                    #cv.imwrite("{0}/d-{1:04d}.jpg".format(imageoutpath, n), cut_c)

            n+=1
            prevCircle = chosen




    cv.imshow("getBall", frame)

    if cv.waitKey(30) & 0xFF == ord('q'): break

videoCapture.release()
cv.destroyAllWindows()