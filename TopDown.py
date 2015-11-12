import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2
import cv2.cv as cv
import numpy as np
import math 
from numpy.linalg import lstsq
from numpy import ones,vstack

##aerial view homography
actualpts = np.zeros([5,2])
actualpts2 = np.zeros([5,2])

actualpts[0][0] = 2112
actualpts[0][1] = 805
actualpts[1][0] = 1816
actualpts[1][1] = 805
actualpts[2][0] = 1816
actualpts[2][1] = 1530
actualpts[3][0] = 2112
actualpts[3][1] = 1530
actualpts[4][0] = 2014
actualpts[4][1] = 1331

actualpts2[0][0] = 743
actualpts2[0][1] = 256
actualpts2[1][0] = 434
actualpts2[1][1] = 266
actualpts2[2][0] = 1105
actualpts2[2][1] = 486
actualpts2[3][0] = 1443
actualpts2[3][1] = 436
actualpts2[4][0] = 1100
actualpts2[4][1] = 381

## Find homography between the views.
size = (4000,4000)
panorama = np.zeros((4000,4000, 3), np.uint8)
(homography, _) = cv2.findHomography(actualpts2, actualpts)



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

      cv2.warpPerspective(frame, homography, size, panorama)
      pitch = cv2.imread("righttop.jpg") 
      for i in range(0,len(good_new)):
        x = int(good_new[i,0])
        y = int(good_new[i,1])
        #cv2.rectangle(frame, (x-w/2,y-h/2), (x+w/2,y+h/2), cv.RGB(0, 0, 255), 1)

      for i in range(0,len(good_old)):
        x = int(good_old[i,0])
        y = int(good_old[i,1])
        ##find topdown rectangle location
        hm = homography
        #(h0,h1,h2)    (x)   (h0*x + h1*y + h2)   (tx)
        #(h3,h4,h5)  * (y) = (h3*x + h4*y + h5) = (ty)
        tz = hm[2,0]* x  + hm[2,1]*y + hm[2,2]
        tx = hm[0,0]* x  + hm[0,1]*y + hm[0,2]
        ty =  hm[1,0]* x  + hm[1,1]*y + hm[1,2]
        tx = int(tx/tz)
        ty = int(ty/tz)
        if(i==20):
          cv2.circle(pitch, (tx,ty),25, cv.RGB(100, 225, 0), thickness=5, lineType=8, shift=0)
          cv2.rectangle(pitch, (tx-w/2,ty-h/2), (tx+w/2,ty+h/2), cv.RGB(255, 0, 0), 1)
          cv2.circle(panorama, (tx,ty),25, cv.RGB(100, 225, 0), thickness=5, lineType=8, shift=0)
          cv2.rectangle(panorama, (tx-w/2,ty-h/2), (tx+w/2,ty+h/2), cv.RGB(255, 0, 0), 1)
          ##find nearest point
          for j in range(0,len(good_old)):
            x2 = int(good_old[j,0])
            y2 = int(good_old[j,1])
            tz2 = hm[2,0]* x2  + hm[2,1]*y2 + hm[2,2]
            tx2 = hm[0,0]* x2  + hm[0,1]*y2 + hm[0,2]
            ty2 =  hm[1,0]* x2  + hm[1,1]*y2 + hm[1,2]
            tx2 = int(tx2/tz2)
            ty2 = int(ty2/tz2)
            sq1 = (tx-tx2)*(tx-tx2)
            sq2 = (ty-ty2)*(ty-ty2)

            #eliminate node that is connected to origin via high intensity path
            #distance
            if(math.sqrt(sq1 + sq2) < 150 and j!=i):
              #trace line:
              points = [(tx,ty),(tx2,ty2)]
              x_coords, y_coords = zip(*points)
              A = vstack([x_coords,ones(len(x_coords))]).T
              m, c = lstsq(A, y_coords)[0]

             
              count = 1
              totalCount = 0.0
              highCount = 0.0

##remove node if ratio is high  
              
##              if(ty2 > ty and tx2 > tx):
##                ty3 = ty2 - 1
##                while(ty3 < ty2):
##                  tx3 = tx + count
##                  ty3 = int(m*(tx3) + c)
##                  #cv2.circle(panorama, (tx3,ty3),3, cv.RGB(100, 30, 60), thickness=4, lineType=8, shift=0)
##                  if(panorama[ty3][tx3][0] > 0 and panorama[ty3][tx3][1]>0):
##                    highCount = highCount + 1.0      
##                  count = count + 1.0
##                  totalCount = totalCount + 1.0
##                print "ratio"
##                print float(highCount/totalCount)

##              if(ty2 < ty and tx2 < tx):
##                ty3 = ty - 1
##                while(ty3 < ty):
##                  tx3 = tx2 + count
##                  ty3 = int(m*(tx3) + c)
##                  #cv2.circle(panorama, (tx3,ty3),3, cv.RGB(100, 30, 60), thickness=4, lineType=8, shift=0)
##                  if(panorama[ty3][tx3][0] > 0 and panorama[ty3][tx3][1]>0):
##                    highCount = highCount + 1.0      
##                  count = count + 1
##                  totalCount = totalCount + 1.0
##                print "ratio"
##                print float(highCount/totalCount)
               
              if(ty2 < ty and tx2 > tx):
                ty3 = ty2 + 1
                while(ty3 > ty2):
                  tx3 = tx + count
                  ty3 = int(m*(tx3) + c)
                  ##cv2.circle(panorama, (tx3,ty3),3, cv.RGB(100, 30, 60), thickness=4, lineType=8, shift=0)
                  if(panorama[ty3][tx3][0] > 0 and panorama[ty3][tx3][1]>0):
                    highCount = highCount + 1      
                  count = count + 1
                  totalCount = totalCount + 1.0
                print "ratio"
                print float(highCount/totalCount)
##            
##              
##              if(ty2 > ty and tx2 < tx):
##                ty3 = ty + 1
##                while(ty3 > ty):
##                  tx3 = tx2 + count
##                  ty3 = int(m*(tx3) + c)
##                  #cv2.circle(panorama, (tx3,ty3),3, cv.RGB(100, 30, 60), thickness=4, lineType=8, shift=0)
##                  if(panorama[ty3][tx3][0] > 0 and panorama[ty3][tx3][1]>0):
##                    highCount = highCount + 1      
##                  count = count + 1
##                  totalCount = totalCount + 1.0
##                print "ratio"
##                print float(highCount/totalCount)
##         
                  
                
              #cv2.line(panorama, (tx2,ty2), (tx,ty), (100,3,4),thickness=5)
              cv2.circle(pitch, (tx2,ty2),25, cv.RGB(100, 30, 0), thickness=5, lineType=8, shift=0)
              cv2.rectangle(pitch, (tx2-w/2,ty2-h/2), (tx2+w/2,ty2+h/2), cv.RGB(255, 0, 0), 1)
              cv2.circle(panorama, (tx2,ty2),25, cv.RGB(100, 30, 0), thickness=5, lineType=8, shift=0)
              cv2.rectangle(panorama, (tx2-w/2,ty2-h/2), (tx2+w/2,ty2+h/2), cv.RGB(255, 0, 0), 1)
            
        else:
          pass
##          cv2.circle(pitch, (tx,ty),25, cv.RGB(255, 255, 0), thickness=5, lineType=8, shift=0)
##          cv2.rectangle(pitch, (tx-w/2,ty-h/2), (tx+w/2,ty+h/2), cv.RGB(255, 0, 0), 1)
##          cv2.circle(panorama, (tx,ty),25, cv.RGB(255, 255, 0), thickness=5, lineType=8, shift=0)
##          cv2.rectangle(panorama, (tx-w/2,ty-h/2), (tx+w/2,ty+h/2), cv.RGB(255, 0, 0), 1)
      
      
      cv2.imshow('frame', panorama)
      cv2.imwrite("background.jpg", panorama)
      cv2.imwrite("pitch.jpg", pitch)
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



