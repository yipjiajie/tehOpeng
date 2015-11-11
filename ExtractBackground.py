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

# for fr in range(0,frameCount-1):
for fr in range(0,60):
    _, frame = cap.read()
    
    if(fr < 50):
      continue

    fgmask1 = fgbg1.apply(frame)
    # fgmask2 = fgbg2.apply(frame)
    # print len(frame[:,:,0])
    # print len(fgmask1/255.0)
    
    # mFrame = np.array(frame[:,:,0]).reshape(frameWidth, frameHeight)
    # mFgMask = np.array(fgmask1).reshape(frameWidth, frameHeight)

    # print len(mFrame), len(mFrame[1,:]), mFrame.size
    # print len(mFgMask), len(mFgMask[1,:]), mFgMask.size

    normalizedMask = 1.0 - fgmask1/255.0


    for i in range(100, 120):
      print normalizedMask[i,:]

    masked = np.ma.array(frame, 
    	mask=np.concatenate((normalizedMask,normalizedMask,normalizedMask)),
    	fill_value=9999)
    # masked = np.ma.masked_array(frame)
    # masked[normalizedMask > 0.0] = np.ma.masked
    # rgbMask = np.zeros([frameWidth, frameHeight, 3])
    # rgbMask[:,:,0] = normalizedMask
    # rgbMask[:,:,1] = normalizedMask
    # rgbMask[:,:,2] = normalizedMask
    # r = np.cross(mFgMask/255.0, mFrame)
    # g = np.cross(mFgMask/255.0, mFrame)
    # b = np.cross(mFgMask/255.0, mFrame)
    # r = normalizedMask * frame[:,:,0]
    # g= normalizedMask * frame[:,:,1]
    # b = normalizedMask * frame[:,:,2]
    # print r
    # result = np.zeros([frameWidth, frameHeight, 3])
    # result[:,:,0] = r
    # result[:,:,1] = g
    # result[:,:,2] = b
    cv2.imshow('frame',masked)
    cv2.waitKey(1)


# for i in range(0,frameWidth-1):
# 	print fgmask2[i,:]

# print fgmask2
# _, img = cap.read()
# avgImg = np.float32(img)
# loopCount = 0;
# frameCountFloat = frameCount * 1.0;
# for fr in range(1,frameCount):
#     _, img = cap.read()

#     avgImg = (fr/ (fr + 1.0))*avgImg + (1.0/(fr + 1.0))*img
#     normImg = cv2.convertScaleAbs(avgImg)

#     precentageCount = int((fr/frameCountFloat) * 100);
#     if(precentageCount > loopCount):
#     	print precentageCount, "%"
#     	loopCount = precentageCount


# convert into uint8 image 
# cv2.imshow('img',img)
# normImg = cv2.convertScaleAbs(avgImg)

# cv2.imwrite(fileName+"_background.jpg", normImg)

cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()



