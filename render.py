import cv2
import numpy as np
import cv2.cv as cv
import utils
import math
from numpy import linalg

test = cv2.imread("final0.jpg")
frame_height= len(test)
frame_width= len(test[0])

print "frame height is ", frame_height
print "frame width is ", frame_width

video =cv2.VideoWriter("000.avi", -1, 24, (frame_width/2,frame_height/2))

for i in range (0,24):
	img = cv2.imread("final"+str(i)+".jpg")
	img=cv2.resize(img,(0,0),fx=0.5, fy=0.5)
	video.write(img)
video.release()

cv2.destroyAllWindows()
