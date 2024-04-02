import os
import cv2 as cv
import numpy as np
from tensorflow import keras
from keras.models import load_model

# Load your trained model
model = load_model('../BumpSetCut/my_model.keras')

#Video stuff 
videoCapture = cv.VideoCapture('../BumpSetCut/video/longerClip.mp4/')
videoCapture.set(cv.CAP_PROP_BUFFERSIZE, 2)
prevCircle = None
dist = lambda x1,y1,x2,y2: (x1-x2)**2+(y1-y2)**2
backSub = cv.createBackgroundSubtractorKNN()
n=0

#Trajectory 
trajectory = []
max_trajectory_length = 20  # Number of points to keep in the trajectory
firstBall = False


#Preprocessing images for model
def preprocess_for_model(img, size=(224, 224)):
    img = cv.resize(img, size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Model expects batch dimension
    return img

#Center of square for trajectory tracking
def get_centroid(x, y, w, h):
    return (int(x + w/2), int(y + h/2))


while True:
    #OpenCV stuff
    ret, frame = videoCapture.read()
    if not ret: break
    frame = cv.resize(frame, (1920,1080))

    #Masks for circle detection
    mask = backSub.apply(frame)
    mask = cv.GaussianBlur(mask, (13, 13),0)
    ret,mask = cv.threshold(mask,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)

    #Circle Detection
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
            ry = chosen[1] - chosen[2]
            rw = chosen[2]*2
            rh = chosen[2]*2
            
            if prevCircle is not None:
                prevX = prevCircle[0] - prevCircle[2]
                prevY = prevCircle[1] - prevCircle[2]
                prevW = prevCircle[2]*2
                prevH = prevCircle[2]*2
            else:
    # Set default values or handle the case where prevCircle is None
                prevX, prevY, prevW, prevH = 0, 0, 0, 0 
            #checking if coordinates are in frame
            height, width = frame.shape[:2]
            x = max(0, rx)
            y = max(0, ry)
            w = min(rw, width - rx)
            h = min(rh, height - ry)

            px = max(0, prevX)
            py = max(0, prevY)
            pw = min(prevW, width - prevX)
            ph = min(prevH, height - prevY)
            
            #rectangle shows what it is cutting for the classifier
            #cv.rectangle(frame, (x-5, y-5), (x + w + 5, y + h + 5), (0, 255, 0), 2)

            #cutting images from black and white mask
            cut_m = mask[y-5:y+h+5, x-5:x+w+5]
            #cutting images from color mask
            cut_f = frame[y-5:y+h+5, x-5:x+w+5]

            #Splicing the images to cutout the background but keep the circle detected in
            cut_c = cv.bitwise_and(cut_f,cut_f,mask = cut_m)

            # Sometimes the frames return a null value so this checks it and will not process null image as it will crash the program
            if cut_c is not None:
                # Preprocess the cutout for the model
                preprocessed_img = preprocess_for_model(cut_c)

                # Use the model to predict
                prediction = model.predict(preprocessed_img)
                predicted_class = 'notBall' if prediction[0][0] > 0.5 else 'ball'
                #print(f"Detected: {predicted_class} with confidence {prediction[0][0]}")

                #First ball detection 

                # * if ball is detected a tracking frame is created, if the next ball is detected in that frame then we have something

                if firstBall and predicted_class == 'ball':

                    track_window = (x - 20, y - 20, w + 40, h + 40)
                    tx, ty, tw, th = track_window
                    #Create tracking window for next ball to be detected
                elif firstBall == False:
                    track_window = (px - 20, py - 20, pw + 40, ph + 40)
                    tx, ty, tw, th = track_window
                    cv.rectangle(frame, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 2)

                     
                if predicted_class == 'ball':

                    #if ball is in track window otherwise remove
                    
                    #Green circle of predicted ball
                    #cv.circle(frame, (chosen[0], chosen[1]), chosen[2], (0,255,0), 3)

                    
                    center = get_centroid(x, y, w, h)
                    trajectory.append(center)
                    # Limit the length of the trajectory
                    if len(trajectory) > max_trajectory_length:
                        trajectory.pop(0)

                    # Draw the trajectory
                    for i in range(1, len(trajectory)):
                        cv.line(frame, trajectory[i - 1], trajectory[i], (0, 255, 0), 2)
                    


                elif predicted_class == 'notBall':
                    cv.circle(frame, (chosen[0], chosen[1]), chosen[2], (0,0,255), 3)
                    print("notball")
            n+=1
            prevCircle = chosen
    
    else:
        print("null")

    cv.imshow("getBall", frame)

    if cv.waitKey(30) & 0xFF == ord('q'): break

videoCapture.release()
cv.destroyAllWindows()