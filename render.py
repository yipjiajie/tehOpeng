import cv2
import numpy as np
import cv2.cv as cv
import math
import time


# test = cv2.imread("final0.jpg")
# frame_height= len(test)
# frame_width= len(test[0])

# print "frame height is ", 486
# print "frame width is ", 3668

#3668 x 1328
video =cv2.VideoWriter("full.avi", cv.CV_FOURCC('M','J','P','G'), 24, (3668,1328))

fullStartTime = time.time()
for i in range (3600,3840):
	# startTime = time.time()

	print "Frame",i

	img = cv2.imread("enlarge/enlarged"+`i`+".jpg")

	# img=cv2.resize(img,(0,0),fx=0.5, fy=0.5)
	frame_height= len(img)
	frame_width= len(img[0])
	
	video.write(img)
	# print "finished: ", time.time() - startTime, "(sec)"


video.release()

print "total finished: ", time.time() - fullStartTime, "(sec)"

cv2.destroyAllWindows()
