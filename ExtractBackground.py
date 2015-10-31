import cv2
import cv2.cv as cv
import numpy as np

print "Enter the filename of the video (w/o extension)"
print "to have its background extracted:"

fileName = raw_input()

cap = cv2.VideoCapture(fileName + ".mp4")

frameHeight = int(cap.get(cv.CV_CAP_PROP_FRAME_WIDTH))
frameWidth = int(cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
frameCount = int(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))

print "Height:",frameHeight
print "Width:",frameWidth
print "Frame Count:", frameCount

_, img = cap.read()
avgImg = np.float32(img)

loopCount = 0;
frameCountFloat = frameCount * 1.0;
for fr in range(1,frameCount):
    _, img = cap.read()

    avgImg = (fr/ (fr + 1.0))*avgImg + (1.0/(fr + 1.0))*img
    normImg = cv2.convertScaleAbs(avgImg)

    precentageCount = int((fr/frameCountFloat) * 100);
    if(precentageCount > loopCount):
    	print precentageCount, "%"
    	loopCount = precentageCount


# convert into uint8 image cv2.imshow('img',img)
normImg = cv2.convertScaleAbs(avgImg)

cv2.imwrite(fileName+"_background.jpg", normImg)

cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()



