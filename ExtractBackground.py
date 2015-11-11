import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2
import cv2.cv as cv
import numpy as np

print "Enter the filename of the video (w/o extension)"
print "to have its background extracted:"

# fileName = raw_input()

# cap = cv2.VideoCapture(fileName + ".mp4")
cap = cv2.VideoCapture("football_mid.mp4")

frameHeight = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
frameWidth = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
frameCount = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))

print "Height:",frameHeight
print "Width:",frameWidth
print "Frame Count:", frameCount


fgbg1 = cv2.BackgroundSubtractorMOG()
fgbg2 = cv2.BackgroundSubtractorMOG2()

_, frame = cap.read()

for fr in range(50,frameCount-1):
# for fr in range(0,60):
    _, frame = cap.read()
    
    # if(fr < 50):
    #   continue

    fgmask1 = fgbg1.apply(frame, learningRate=0.001)
    # fgmask2 = fgbg2.apply(frame, learningRate=0.001)

    normalizedMask = fgmask1/255.0

    result = frame;
    result[:,:,0] = frame[:,:,0] * normalizedMask
    result[:,:,1] = frame[:,:,1] * normalizedMask
    result[:,:,2] = frame[:,:,2] * normalizedMask
    cv2.imshow('frame', result)
    cv2.waitKey(1)


# convert into uint8 image 
# cv2.imshow('img',img)
# normImg = cv2.convertScaleAbs(avgImg)

# cv2.imwrite(fileName+"_background.jpg", normImg)

cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()



