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
hits =0
miss =0
kernel_e = np.ones((10,10),np.uint8)
kernel_d = np.ones((5,5),np.uint8)
pxl_sum =0
pxl =0
frame_cnt =0
dur_cnt =0


while True:

	ret,frame_full = video.read()

	if frame_full is None:
		break

	frame_cnt+=1
	# frame = frame_full[230:490,550:720]
	frame = frame_full[230:430,450:800]
	frame_full = imutils.resize(frame_full,width=700,height=700)
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (25,25),0)
	
	if first_frame is None:
		first_frame = gray
		continue

	frame_diff = cv2.absdiff(first_frame,gray)
	t,thresh = cv2.threshold(frame_diff,20,255,cv2.THRESH_BINARY)
	# thresh = cv2.GaussianBlur(thresh, (25,25),0)
	thresh = cv2.erode(thresh,kernel_e,iterations = 1)
	thresh = cv2.dilate(thresh,kernel_e, iterations=1)

	thresh = cv2.Canny(thresh,20,255)
	thresh = cv2.dilate(thresh,kernel_d, iterations=1)

	pxl_n = len(thresh[thresh==255])
	if pxl_n >500:
		pxl_sum +=pxl_n
		pxl =1
	elif pxl_n <1:
		pxl =0

	if pxl ==0: 
		if pxl_sum >35000 and (frame_cnt-dur_cnt)>100:
			hits+=1
			dur_cnt = frame_cnt
		elif pxl_sum <35000 and pxl_sum >15000 and  (frame_cnt-dur_cnt)>100:
			miss +=1
			dur_cnt =frame_cnt
		pxl_sum =0


	cv2.putText(frame_full, "Hits: {}".format(hits), (20,380), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
	cv2.putText(frame_full, "Miss: {}".format(miss), (90,380), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
	cv2.imshow("Security Feed", frame_full)
	cv2.imshow("Thresh", thresh)
	cv2.waitKey(1)& 0xFF


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


