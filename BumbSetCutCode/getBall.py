import os
import cv2 as cv
import numpy as np



#this is the path of where getball outputs its images into

imageoutpath = ('../BumpSetCut/images/getballimages')
videopath = ("../BumpSetCut/video/Rangle.mp4")
#This is where you set the path to the volleyball video

videoCapture = cv.VideoCapture(videopath)
videoCapture.set(cv.CAP_PROP_BUFFERSIZE, 2)
prevCircle = None
dist = lambda x1,y1,x2,y2: (x1-x2)**2+(y1-y2)**2

backSub = cv.createBackgroundSubtractorKNN()
n=0
while True:
    ret, frame = videoCapture.read()
    if not ret: break
    
    frame = cv.resize(frame, (1280,720))

    mask = backSub.apply(frame)


    mask = cv.GaussianBlur(mask, (13, 13),0)
    ret,mask = cv.threshold(mask,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)


    #See for HoughCircles perameter description https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
    circles = cv.HoughCircles(mask, cv.HOUGH_GRADIENT_ALT, 1.5, 1, param1=250, param2= 0.85, minRadius=3, maxRadius=40)

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


            #These draw the circles being detected in each frame 
            cv.circle(frame, (chosen[0], chosen[1]), 1, (0,100,100), 3)
            cv.circle(frame, (chosen[0], chosen[1]), chosen[2], (255,0,255), 3)

            #Sometimes the frames return a null value so this checks it and will not write null image as it will crash the program
            if cut_c is None:
                 print("none")
            else:
                cv.imwrite("{0}/d-{1:04d}.jpg".format(imageoutpath, n), cut_c)

            n+=1
            prevCircle = chosen




    cv.imshow("getBall", frame)

    if cv.waitKey(1) & 0xFF == ord('q'): break

videoCapture.release()
cv.destroyAllWindows()