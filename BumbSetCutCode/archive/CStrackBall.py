import os
import cv2 as cv
import numpy as np
from tensorflow import keras
from keras.models import load_model

# Load your trained model
model = load_model('../BumpSetCut/my_model.keras')


#this is the path of where getball outputs its images into
imageoutpath = ('../BumpSetCut/images')


#This is where you set the path to the volleyball video
videoCapture = cv.VideoCapture('../BumpSetCut/video/shorterClip.mp4/')
videoCapture.set(cv.CAP_PROP_BUFFERSIZE, 2)
prevCircle = None
dist = lambda x1,y1,x2,y2: (x1-x2)**2+(y1-y2)**2

backSub = cv.createBackgroundSubtractorKNN()
n=0

# Initialize tracking variables
ball_detected = False
track_window = None
roi_hist = None



def preprocess_for_model(img, size=(224, 224)):
    img = cv.resize(img, size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Model expects batch dimension
    return img

def get_centroid(x, y, w, h):
    return (int(x + w/2), int(y + h/2))




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
            ry = chosen[1] - chosen[2]
            rw = chosen[2]*2
            rh = chosen[2]*2


            
            #rectangle shows what it is cutting for the classifier
            #cv.rectangle(frame,((rx-5), ry+5), ((rx+rw+5),(ry-rh-5)),(255,0,255), 3)
            #print(chosen[2])

            height, width = frame.shape[:2]
            x = max(0, rx)
            y = max(0, ry)
            w = min(rw, width - rx)
            h = min(rh, height - ry)

            #cv.rectangle(frame, (x-5, y-5), (x + w + 5, y + h + 5), (0, 255, 0), 2)  # Green rectangle with a thickness of 2

            #cutting images from black and white mask
            cut_m = mask[y-5:y+h+5, x-5:x+w+5]

            #cutting images from color mask
            cut_f = frame[y-5:y+h+5, x-5:x+w+5]

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

                #first ball detection to set roi up
                if not ball_detected and predicted_class == 'ball':
                    #first two consecutive ball detectyions? start with one
                    ball_detected = True
                    track_window = (rx - 5, ry - rh - 5, rw + 5, rh + 10) #initial tracking window

                    #set up ROI for tracking
                    roi = frame[y-5:y+h+5, x-5:x+w+5]
                    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

                    # Create a mask and calculate the histogram for backprojection
                    mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
                    roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
                    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

                    # CAMShift tracking
                    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
                    
                if ball_detected:
                    ball_detected = True
                    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                    dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    
                    # Apply CAMShift to get the new location
                    ret, track_window = cv.CamShift(dst, track_window, term_crit)

                    # Draw tracking window on the frame
                    pts = cv.boxPoints(ret)
                    pts = np.int0(pts)
                    frame = cv.polylines(frame, [pts], True, 255, 2)
                    
                    
                    cv.circle(frame, (chosen[0], chosen[1]), chosen[2], (0,255,0), 3)
                    #rectangle below
                    #top_left = (rx, (ry - 2*chosen[2]))
                    #bottom_right = (rx + rw, (ry - 2*chosen[2]) + rh)
                    #cv.rectangle(frame, top_left, bottom_right, (0, 255, 0), 3) 
                    x, y, w, h = track_window
                    if w * h > 200 or w * h < 20:
                        ball_detected = False  # Reset tracking if window size is too big/small
                    else:
                        pts = cv.boxPoints(ret)
                        pts = np.int0(pts)
                        frame = cv.polylines(frame, [pts], True, 255, 2)

                elif predicted_class == 'notBall':

                    ball_detected = False
                    #cv.circle(frame, (chosen[0], chosen[1]), 1, (255,0,0), 3)
                    cv.circle(frame, (chosen[0], chosen[1]), chosen[2], (0,0,255), 3)
                    #cv.imwrite("{0}/d-{1:04d}.jpg".format(imageoutpath, n), cut_c)

            n+=1
            prevCircle = chosen




    cv.imshow("getBall", frame)

    if cv.waitKey(30) & 0xFF == ord('q'): break

videoCapture.release()
cv.destroyAllWindows()