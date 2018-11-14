import cv2
import numpy as np
import sys
import imutils
import matplotlib.pyplot as plt

path = 'SourceClip.mp4'
video = cv2.VideoCapture(path)

if not video.isOpened():
	print("Video could not be opened")
	sys.exit()
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frame_rate = video.get(cv2.CAP_PROP_FPS)
print(num_frames)
print(frame_rate)

# ret,frame = video.read()
# # frame = frame[230:500,450:800]
# kernel = np.ones((5,5),np.uint8)
# gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

# blank = np.zeros(gray.shape)

# thresh = cv2.erode(gray,kernel,iterations = 2)
# edges = cv2.Canny(thresh,100,200)

# cv2.imshow("Frame",edges)
# cv2.waitKey()


first_frame = None
writer = None
hit =0
kernel = np.ones((10,10),np.uint8)
while True:

	ret,frame = video.read()

	if frame is None:
		break
	frame = frame[230:500,450:800]
	# frame = frame[320:390,580:700]
	# frame = imutils.resize(frame,width=500,height=500)
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (25,25),0)
	# edges = cv2.Canny(gray,20,255)
	
	if first_frame is None:
		first_frame = gray
		continue

	frame_diff = cv2.absdiff(first_frame,gray)
	t,thresh = cv2.threshold(frame_diff,5,255,cv2.THRESH_BINARY)

	thresh = cv2.erode(thresh,kernel,iterations = 2)
	thresh = cv2.dilate(thresh,None, iterations=2)
	thresh = cv2.Canny(thresh,20,255)
	# thresh = cv2.dilate(thresh,None, iterations=2)
	# if len(thresh ==255) >200:
	# 	hit+=1
	cv2.imshow("Security Feed", frame)
	cv2.imshow("Thresh", thresh)
	cv2.waitKey(1)& 0xFF


print(hit)
video.release()
cv2.destroyAllWindows()


# frame = frame[230:500,450:800]
# # imutils.resize(frame,width=600)
# cv2.imshow('First_Frame',frame)
# cv2.waitKey()
# plt.imshow(frame)
# plt.show()
# 	if frame is None:
# 		break
# 	frame = imutils.resize(frame,width=600)
# 	blurred = cv2.GaussianBlur(frame, (11,11),0)
# 	hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
	# gray = cv2.cvtColor(first_frame,cv2.COLOR_BGR2GRAY)

# circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1.2,100)

# if circles is not None:
# 	circles = np.round(circles[0,:]).astype("int")

# 	for x,y,r in circles:
# 		cv2.circle(gray,(x,y),r,(0,255,0),4)
# ret,frame = video.read()

# frame = imutils.resize(frame,width=700)
# gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
# blurred = cv2.medianBlur(gray,5)
# # hsv = cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
# circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1.2,200)

# # if circles is not None:
# 	# circles = np.round(circles[0,:]).astype("int")

# # for (x,y,r) in circles:
# # 	cv2.circle(blurred,(x,y),r,(0,255,0),4)
# # # mask = cv2.inRange(hsv, Lower, Upper)
# # # mask = cv2.erode(mask, None, iterations=2)
# # # mask = cv2.dilate(mask, None, iterations=2)	
# cv2.imshow('First_Frame',gray)
# cv2.waitKey()


