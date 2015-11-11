import cv2
import numpy as np
import cv2.cv as cv
import utils
import math
from numpy import linalg

def findDimensions( image, homography):
	base_p1 = np.ones(3, np.float32)
	base_p2 = np.ones(3, np.float32)
	base_p3 = np.ones(3, np.float32)
	base_p4 = np.ones(3, np.float32)
    
	(y, x) = image.shape[:2]
    
	base_p1[:2] = [0,0]
	base_p2[:2] = [x,0]
	base_p3[:2] = [0,y]
	base_p4[:2] = [x,y]
    
	max_x = None
	max_y = None
	min_x = None
	min_y = None
    
	for pt in [base_p1, base_p2, base_p3, base_p4]:
			
		hp = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T
		
		hp_arr = np.array(hp, np.float32)
    
		normal_pt = np.array([hp_arr[0]/hp_arr[2], hp_arr[1]/hp_arr[2]], np.float32)
    
		if ( max_x == None or normal_pt[0,0] > max_x ):
			max_x = normal_pt[0,0]
    
		if ( max_y == None or normal_pt[1,0] > max_y ):
			max_y = normal_pt[1,0]
    
		if ( min_x == None or normal_pt[0,0] < min_x ):
			min_x = normal_pt[0,0]
    
		if ( min_y == None or normal_pt[1,0] < min_y ):
			min_y = normal_pt[1,0]
    
	min_x = min(0, min_x)
	min_y = min(0, min_y)
		
	return (min_x, min_y, max_x, max_y)

def stitch(base_frame,to_be_stitched_frame,H,status):
	
		
	inlierRatio = float(np.sum(status)) / float(len(status))
	
	H = H / H[2,2]
	H_inv = linalg.inv(H)

	if ( inlierRatio > 0.1 ):
		(min_x, min_y, max_x, max_y)=findDimensions(to_be_stitched_frame, H_inv)
		 # Adjust max_x and max_y by base img size
		max_x = max(max_x, base_frame.shape[1])
		max_y = max(max_y, base_frame.shape[0])
	
		move_h = np.matrix(np.identity(3), np.float32)
		if ( min_x < 0 ):
			move_h[0,2] += -min_x
			max_x += -min_x
    
		if ( min_y < 0 ):
			move_h[1,2] += -min_y
			max_y += -min_y
		#print "Inverse Homography: \n", H_inv
		#print "Min Points: ", (min_x, min_y)
	
		mod_inv_h = move_h * H_inv
		
		img_w = int(math.ceil(max_x))
		img_h = int(math.ceil(max_y))
		
		#print "New Dimensions: ", (img_w, img_h)
		
		# stitch
		base_img_warp = cv2.warpPerspective(base_frame, move_h, (img_w, img_h))
		to_be_stitched_warp = cv2.warpPerspective(to_be_stitched_frame, mod_inv_h, (img_w, img_h))

		enlarged_base_img = np.zeros((img_h, img_w, 3), np.uint8)
		#print "Enlarged Image Shape: ", enlarged_base_img.shape
		#print "Base Image Shape: ", base_frame.shape
		#print "Base Image Warp Shape: ", base_frame.shape
		(ret,data_map) = cv2.threshold(cv2.cvtColor(to_be_stitched_warp, cv2. COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
		
		enlarged_base_img = cv2.add(enlarged_base_img, base_img_warp, mask=np.bitwise_not(data_map), dtype=cv2.CV_8U)
		
		# Now add the warped image
		final_img = cv2.add(enlarged_base_img, to_be_stitched_warp, dtype=cv2.CV_8U)
		
		# Crop off the black edges
		final_gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
		_, thresh = cv2.threshold(final_gray, 1, 255, cv2.THRESH_BINARY)
		contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		#print "Found %d contours..." % (len(contours))
		
		max_area = 0
		best_rect = (0,0,0,0)
		for cnt in contours:
			x,y,w,h = cv2.boundingRect(cnt)
			# print "Bounding Rectangle: ", (x,y,w,h)
    
			deltaHeight = h-y
			deltaWidth = w-x
			
			area = deltaHeight * deltaWidth
			
			if ( area > max_area and deltaHeight > 0 and deltaWidth > 0):
				max_area = area
				best_rect = (x,y,w,h)
			
			if ( max_area > 0 ):
				#print "Maximum Contour: ", max_area
				#print "Best Rectangle: ", best_rect
				
				final_img_crop = final_img[best_rect[1]:best_rect[1]+best_rect[3],best_rect[0]:best_rect[0]+best_rect[2]]
				return final_img_crop
	
left_cap = cv2.VideoCapture("football_left.mp4")
mid_cap = cv2.VideoCapture("football_mid.mp4")
right_cap = cv2.VideoCapture("football_right.mp4")

width= mid_cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT)
height= mid_cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)
fps=mid_cap.get(cv.CV_CAP_PROP_FPS)
frameCount= mid_cap.get(cv.CV_CAP_PROP_FRAME_COUNT)

print "height=",height
print "width=",width
print "fps",fps
print "framecount=",frameCount
print left_cap.get(cv.CV_CAP_PROP_FOURCC)
 #CV_FOURCC('M','S','V','C')
 #828601953
#video  =cv2.VideoWriter("Stitched_video3.avi", -1, 1, (2409,615))
for iterate in range (0,1005):
#for iterate in range (0,1):
	# Capture frame-by-frame
	ret1, left_frame = left_cap.read()
	ret2, mid_frame = mid_cap.read()
	ret3, right_frame = right_cap.read()
#while(1):
for iterate in range (0,1):
	# Capture frame-by-frame
	ret1, left_frame = left_cap.read()
	ret2, mid_frame = mid_cap.read()
	ret3, right_frame = right_cap.read()
	
	# Get gray scale Image
	left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
	mid_gray = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2GRAY)
	right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
	
	#Get Gaussian blur
	left_blur = cv2.GaussianBlur(left_gray, (5,5), 0)
	mid_blur = cv2.GaussianBlur(mid_gray, (5,5), 0)
	right_blur = cv2.GaussianBlur(right_gray, (5,5), 0)
	
	# Use the SIFT feature detector
	detector = cv2.SIFT()
	# Find key points in base image for motion estimation
	left_features, left_descs = detector.detectAndCompute(left_blur, None)
	mid_features, mid_descs = detector.detectAndCompute(mid_blur, None)
	right_features, right_descs = detector.detectAndCompute(right_blur, None)
	
	left_points = []
	mid_points = []
	right_points = []
	for kp in left_features:
		left_points.append((int(kp.pt[0]),int(kp.pt[1])))
	for kp in mid_features:
		mid_points.append((int(kp.pt[0]),int(kp.pt[1])))
	for kp in right_features:
		right_points.append((int(kp.pt[0]),int(kp.pt[1])))
	
	# Parameters for nearest-neighbor matching
	FLANN_INDEX_KDTREE = 1  
	flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	matcher = cv2.FlannBasedMatcher(flann_params, {})
	
	#find left to mid first
	left_to_mid_matches = matcher.knnMatch(left_descs, trainDescriptors=mid_descs, k=2)
	#print "Left-to-Mid Match Count: ", len(left_to_mid_matches)
	#find right to mid next
	right_to_mid_matches = matcher.knnMatch(right_descs, trainDescriptors=mid_descs, k=2)
	#print "Right-to-Mid Match Count: ", len(right_to_mid_matches)
	
	left_to_mid_filtered_matches = []
	right_to_mid_filtered_matches = []
	ratio = 0.75
	
	for m in left_to_mid_matches:
		if len(m) == 2 and m[0].distance < m[1].distance * ratio:
			left_to_mid_filtered_matches.append(m[0])
	left_to_mid_matches_subset = left_to_mid_filtered_matches		
	for m in right_to_mid_matches:
		if len(m) == 2 and m[0].distance < m[1].distance * ratio:
			right_to_mid_filtered_matches.append(m[0])
	right_to_mid_matches_subset = right_to_mid_filtered_matches	
	
	#print "Left-to-Mid Filtered Match Count: ", len(left_to_mid_matches_subset)
	#print "Right-to-Mid Filtered Match Count: ", len(right_to_mid_matches_subset)
	
	left_to_mid_sumDistance = 0.0
	for match in left_to_mid_matches_subset:
		left_to_mid_sumDistance += match.distance
	left_to_mid_distance=left_to_mid_sumDistance
	right_to_mid_sumDistance = 0.0
	for match in right_to_mid_matches_subset:
		right_to_mid_sumDistance += match.distance
	right_to_mid_distance=right_to_mid_sumDistance
	
	#print "Left-to-Mid Distance: ", left_to_mid_distance
	#print "right-to-Mid Distance: ", right_to_mid_distance
	
	left_to_mid_averagePointDistance = left_to_mid_distance/float(len(left_to_mid_matches_subset))
	right_to_mid_averagePointDistance = right_to_mid_distance/float(len(right_to_mid_matches_subset))
	
	#print "Left-to-Mid Average Distance: ", left_to_mid_averagePointDistance
	#print "Right-to-Mid Average Distance: ", right_to_mid_averagePointDistance
	
	left_kp = []
	mid_kp = []
	
	#print "left_features size: ", len(left_features)
	#print "mid_features size: ", len(mid_features)
	#print "right_features size: ", len(right_features)
	for match in left_to_mid_matches_subset:
		mid_kp.append(mid_features[match.trainIdx])
		left_kp.append(left_features[match.queryIdx])
    
	p1 = np.array([k.pt for k in mid_kp])
	p2 = np.array([k.pt for k in left_kp])

	H_left, status_left = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
	left_half_frame=stitch(mid_frame,left_frame,H_left,status_left)
	
	#cv2.imwrite("left.jpg", left_half_frame)
	
	mid2_kp = []
	right_kp = []
	
	for match in right_to_mid_matches_subset:
		mid2_kp.append(mid_features[match.trainIdx])
		right_kp.append(right_features[match.queryIdx])
	
	p1 = np.array([k.pt for k in mid2_kp])
	p2 = np.array([k.pt for k in right_kp])
	H_right, status_right = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
	right_half_frame=stitch(mid_frame,right_frame,H_right,status_right)
	
	#cv2.imwrite("right.jpg", right_half_frame)
	
	#utils.showImage(right_half_frame, scale=(0.2, 0.2), timeout=0)
	#utils.showImage(left_half_frame, scale=(0.2, 0.2), timeout=0)
	#new_gray = cv2.cvtColor(left_half_frame, cv2.COLOR_BGR2GRAY)
	#new_blur = cv2.GaussianBlur(new_gray, (5,5), 0)
	
	left_src = left_half_frame
	right_src = right_half_frame

	left_small_image = cv2.resize(left_src, (0,0), fx=0.2, fy=0.2) 
	right_small_image = cv2.resize(right_src, (0,0), fx=0.2, fy=0.2) 
	#cv2.imwrite("reducedleft.jpg", left_small_image)
	#cv2.imwrite("reducedright.jpg", right_small_image)
	# Use the SIFT feature detector
	detector2 = cv2.SIFT()
	# Find key points in base image for motion estimation
	new_left_gray = cv2.cvtColor(left_small_image, cv2.COLOR_BGR2GRAY)
	new_right_gray = cv2.cvtColor(right_small_image, cv2.COLOR_BGR2GRAY)
	#blur image
	new_left_blur = cv2.GaussianBlur(new_left_gray, (5,5), 0)
	new_right_blur = cv2.GaussianBlur(new_right_gray, (5,5), 0)
	# Find key points in base image for motion estimation
	new_left_features, new_left_descs = detector.detectAndCompute(new_left_blur, None)
	new_right_features, new_right_descs = detector.detectAndCompute(new_right_blur, None)
	left_to_right_matches = matcher.knnMatch(new_left_descs, trainDescriptors=new_right_descs, k=2)
	
	new_left_kp = []
	new_right_kp = []
	
	left_to_right_filtered_matches = []
	ratio = 0.75
	for m in left_to_right_matches:
		if len(m) == 2 and m[0].distance < m[1].distance * ratio:
			left_to_right_filtered_matches.append(m[0])
	left_to_right_matches_subset = left_to_right_filtered_matches
	for match in left_to_right_matches_subset:
		new_right_kp.append(new_right_features[match.trainIdx])
		new_left_kp.append(new_left_features[match.queryIdx])
    
	p1 = np.array([k.pt for k in new_left_kp])
	p2 = np.array([k.pt for k in new_right_kp])
	
	H_final, status_final = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
	#print '%d / %d  inliers/matched' % (np.sum(status), len(status))

	final_frame=stitch(left_small_image,right_small_image,H_final,status_final)
	#video.write(final_frame)
	
	cv2.imwrite("final"+str(iterate)+".jpg", final_frame)
	finalheight=len(final_frame)
	finalwidth=len(final_frame[0])
	print "height", len(final_frame)
	print "width", len(final_frame[0])
	#utils.showImage(final_frame, scale=(0.5, 0.5), timeout=0)
	frame_height=len(final_frame)
	frame_width=len(final_frame[0])
	print "height", frame_height
	print "width", frame_width
	half_width=frame_width/2
	top_coord=0
	bottom_coord=0
	count=0
	for i in range (0,frame_height):
		if final_frame[i,half_width][0]!=0:
			if count ==0:
				top_coord=i
				count=1
			bottom_coord=i
			
	print "top coord", top_coord
	print "bottom coord", bottom_coord
		
	cv2.waitKey(1)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	
	#detector=cv2.SIFT()
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
		
final_frame_=final_frame		
video  =cv2.VideoWriter("Stitched_video4.avi", -1, 24, (frame_width,(bottom_coord-top_coord)))
left_cap = cv2.VideoCapture("football_left.mp4")
mid_cap = cv2.VideoCapture("football_mid.mp4")
right_cap = cv2.VideoCapture("football_right.mp4")
for iterate in range (0,7200):
	# Capture frame-by-frame
	ret1, left_frame = left_cap.read()
	ret2, mid_frame = mid_cap.read()
	ret3, right_frame = right_cap.read()
	left_half_frame=stitch(mid_frame,left_frame,H_left,status_left)
	right_half_frame=stitch(mid_frame,right_frame,H_right,status_right)
	left_src = left_half_frame
	right_src = right_half_frame

	left_small_image = cv2.resize(left_src, (0,0), fx=0.2, fy=0.2) 
	right_small_image = cv2.resize(right_src, (0,0), fx=0.2, fy=0.2) 
	#final_frame=(finalwidth,finalheight)
	final_frame=stitch(left_small_image,right_small_image,H_final,status_final)
	
	cropped_final=final_frame[top_coord:bottom_coord,0:frame_width]
	#utils.showImage(final_frame, scale=(0.5, 0.5), timeout=0)
	#utils.showImage(cropped_final, scale=(0.5, 0.5), timeout=0)
		
		
	#if len(final_frame)==finalheight:
	#	if len(final_frame[0])==finalwidth:
	#		video.write(final_frame)
	#		final_frame_=final_frame
	#		print "correct"
	#	else:
	#		video.write(final_frame_)
	#		final_frame_=final_frame
	#else:
	#	video.write(final_frame_)
	video.write(cropped_final)
	print "pass",iterate
#release capture
#cap1.release()
print "End"
video.release()


cv2.destroyAllWindows()
