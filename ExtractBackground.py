import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2
import cv2.cv as cv
import numpy as np

print "Enter the filename of the video (w/o extension)"
print "to have its background extracted:"

# fileName = raw_input()

# cap = cv2.VideoCapture(fileName + ".mp4")
cap = cv2.VideoCapture("football_right.mp4")

frameHeight = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
frameWidth = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
frameCount = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))

print "Height:",frameHeight
print "Width:",frameWidth
print "Frame Count:", frameCount


fgbg1 = cv2.BackgroundSubtractorMOG()
fgbg2 = cv2.BackgroundSubtractorMOG2()

_, frame = cap.read()

# fgmask1 = fgbg1.apply(frame, learningRate=0.001)
fgmask2 = fgbg2.apply(frame, learningRate=0.001)

normalizedMask = fgmask2/255.0

result = frame;
result[:,:,0] = frame[:,:,0] * normalizedMask
result[:,:,1] = frame[:,:,1] * normalizedMask
result[:,:,2] = frame[:,:,2] * normalizedMask

# find corners in the first frame
frame_first = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

corners_first = cv2.goodFeaturesToTrack(frame_first, minDistance=30,
  maxCorners = 500, qualityLevel=0.1, blockSize=10, useHarrisDetector=0, k=0.04)


# w = 50
# h = 50
# for i in range(0,len(corners_first)):
#   x = int(corners_first[i,0,0])
#   y = int(corners_first[i,0,1])
#   cv2.rectangle(frame, (x-w/2,y-h/2), (x+w/2,y+h/2), 255, 2)

# cv2.imshow('frame', frame)
# cv2.waitKey(1)


for fr in range(0,frameCount-1):
# for fr in range(0,60):
    _, frame = cap.read()
    

    if(fr < 250):
      continue

    fgmask1 = fgbg2.apply(frame, learningRate=0.001)
    fgmask2 = fgbg2.apply(frame, learningRate=0.001)

    normalizedMask = fgmask2/255.0

    result = frame;
    result[:,:,0] = frame[:,:,0] * normalizedMask
    result[:,:,1] = frame[:,:,1] * normalizedMask
    result[:,:,2] = frame[:,:,2] * normalizedMask

    # cv2.imshow('frame', result)
    # cv2.waitKey(1)

    frame_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    # calculate optical flow
    corners, st, err = cv2.calcOpticalFlowPyrLK(frame_first, frame_gray, 
      prevPts=corners_first, nextPts=None, maxLevel=3, winSize=(3,21), flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS)

    # Select good points
    good_new = corners[st==1]
    good_old = corners_first[st==1]


    w = 10
    h = 10
    for i in range(0,len(good_new)):
      x = int(good_new[i,0])
      y = int(good_new[i,1])
      cv2.rectangle(frame, (x-w/2,y-h/2), (x+w/2,y+h/2), cv.RGB(0, 255, 0), 1)

    for i in range(0,len(good_old)):
      x = int(good_old[i,0])
      y = int(good_old[i,1])
      cv2.rectangle(frame, (x-w/2,y-h/2), (x+w/2,y+h/2), cv.RGB(255, 0, 0), 1)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)


    # Now update the previous frame and previous points
    frame_first = frame_gray.copy()
    corners_first = good_new.reshape(-1,1,2)


# convert into uint8 image 
# cv2.imshow('img',img)
# normImg = cv2.convertScaleAbs(avgImg)

# cv2.imwrite(fileName+"_background.jpg", normImg)

cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()



