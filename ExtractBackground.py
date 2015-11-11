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

def getForeground(frame):

  # fgmask1 = fgbg1.apply(frame, learningRate=0.01)
  fgmask2 = fgbg2.apply(frame, learningRate=0.01)

  normalizedMask = fgmask2/255.0

  result = frame;
  result[:,:,0] = frame[:,:,0] * normalizedMask
  result[:,:,1] = frame[:,:,1] * normalizedMask
  result[:,:,2] = frame[:,:,2] * normalizedMask

  return result

fgbg1 = cv2.BackgroundSubtractorMOG()
fgbg2 = cv2.BackgroundSubtractorMOG2()

w = 10
h = 10

isFirstFrame = 0
frame_old = 0
corners_old = 0
corners_total = 0
corners_count = 0

print "Skipping First 50 frames"

for fr in range(0,frameCount-1):
# for fr in range(0,60):
    _, frame = cap.read()
    
    result = getForeground(frame)

    if(isFirstFrame < 1 and fr < 50):
      cv2.imshow('frame', result)
      cv2.waitKey(1)
      continue

    if(isFirstFrame < 1 or corners_count < corners_total * 0.95 or corners_count > corners_total * 1.05):

      # find corners in the first frame
      frame_old = cv2.cvtColor(getForeground(frame), cv2.COLOR_BGR2GRAY)
      corners = cv2.goodFeaturesToTrack(frame_old, minDistance=30,
        maxCorners = 500, qualityLevel=0.1, blockSize=10, useHarrisDetector=0, k=0.04)
      corners_old = np.array([c for c in corners if (c[0,1] >= 250 and c[0,1] <= 1000)])
      corners_count = len(corners_old)
      corners_total = corners_count

      isFirstFrame = 1;
      continue
    

    if(isFirstFrame > 0):
      frame_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
      
      # calculate optical flow
      corners, st, err = cv2.calcOpticalFlowPyrLK(frame_old, frame_gray, 
        prevPts=corners_old, nextPts=None, maxLevel=5, winSize=(50,50), flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS)

      # Select good points
      good_new = corners[st==1]
      good_old = corners_old[st==1]


      for i in range(0,len(good_new)):
        x = int(good_new[i,0])
        y = int(good_new[i,1])
        cv2.rectangle(frame, (x-w/2,y-h/2), (x+w/2,y+h/2), cv.RGB(0, 0, 255), 1)

      for i in range(0,len(good_old)):
        x = int(good_old[i,0])
        y = int(good_old[i,1])
        cv2.rectangle(frame, (x-w/2,y-h/2), (x+w/2,y+h/2), cv.RGB(255, 0, 0), 1)

      cv2.imshow('frame', frame)
      cv2.waitKey(1)


      # Now update the previous frame and previous points
      frame_old = frame_gray.copy()
      corners_old = good_new.reshape(-1,1,2)
      corners_count = len(corners_old)


# convert into uint8 image 
# cv2.imshow('img',img)
# normImg = cv2.convertScaleAbs(avgImg)

# cv2.imwrite(fileName+"_background.jpg", normImg)

cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()



