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

# fgmask1 = fgbg1.apply(frame, learningRate=0.001)
fgmask1 = fgbg2.apply(frame, learningRate=0.001)

normalizedMask = fgmask1/255.0

result = frame;
result[:,:,0] = frame[:,:,0] * normalizedMask
result[:,:,1] = frame[:,:,1] * normalizedMask
result[:,:,2] = frame[:,:,2] * normalizedMask

# find corners in the first frame
frame_first = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

corners_first = cv2.goodFeaturesToTrack(fgmask1, minDistance=0.75,
  maxCorners = 200, qualityLevel=0.1, blockSize=3, useHarrisDetector=1, k=0.04)

print corners_first

# Create a mask image for drawing purposes
mask = np.zeros_like(frame_first)

for fr in range(0,frameCount-1):
# for fr in range(0,60):
    _, frame = cap.read()
    
    # skip digital banner
    if(fr < 50):
      continue

    fgmask1 = fgbg2.apply(frame, learningRate=0.001)
    # fgmask2 = fgbg2.apply(frame, learningRate=0.001)

    normalizedMask = fgmask1/255.0

    result = frame;
    result[:,:,0] = frame[:,:,0] * normalizedMask
    result[:,:,1] = frame[:,:,1] * normalizedMask
    result[:,:,2] = frame[:,:,2] * normalizedMask

    cv2.imshow('frame', result)
    cv2.waitKey(1)

    # frame_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    # # calculate optical flow
    # corners, st, err = cv2.calcOpticalFlowPyrLK(frame_first, frame_gray, 
    # 	prevPts=corners_first, nextPts=None)

    # # Select good points
    # good_new = corners_first[st==1]
    # good_old = corners[st==1]

    # # draw the tracks
    # for i,(new,old) in enumerate(zip(good_new,good_old)):
    #     a,b = new.ravel()
    #     c,d = old.ravel()
    #     mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    #     frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    # img = cv2.add(frame,mask)

    # cv2.imshow('frame', img)
    # cv2.waitKey(1)

    # # Now update the previous frame and previous points
    # frame_first = frame_gray.copy()
    # corners_first = good_new.reshape(-1,1,2)


# convert into uint8 image 
# cv2.imshow('img',img)
# normImg = cv2.convertScaleAbs(avgImg)

# cv2.imwrite(fileName+"_background.jpg", normImg)

cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()



