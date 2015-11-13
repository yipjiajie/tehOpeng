import cv2
import numpy as np
import cv2.cv as cv
import utils
import math
from numpy import linalg

test = cv2.imread("final0.jpg")
frame_height= len(test)
frame_width= len(test[0])

print "frame height is ", 486
print "frame width is ", 3668

video =cv2.VideoWriter("000.avi", -1, 24, (3668,486))

for i in range (0,24):
	img = cv2.imread("final"+str(i)+".jpg")
	img=cv2.resize(img,(0,0),fx=0.5, fy=0.5)
	frame_height= len(img)
	frame_width= len(img[0])
	print "pass ", i
	video.write(img)
video.release()

cv2.destroyAllWindows()
