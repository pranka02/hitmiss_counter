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
# print("Number of Frames:"+str(num_frames))
# print("Frame Rate:"+str(frame_rate)+"\n")


first_frame = None
writer = None
hit =0
flag=1

# Parameters
limit_thr=15 #Limit for cv2.threshold
kernel_size=5 #Size of kernel for cv2.erode and cv2.morphologyEx
kernel = np.ones((kernel_size,kernel_size),np.uint8) 
blur_size=9 #Size of Gaussian Blur for cv2.GaussianBlur
median_size=5 #Size of Median Blur for cv2.medianBlur
canny_thr=20 #Limit for cv2.Canny


while True:
	ret,frame_full = video.read()

	if frame_full is None:
		break

	frame = frame_full[230:550,450:800]
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#        gray = cv2.GaussianBlur(gray, (blur_size,blur_size),0)
#        gray = cv2.medianBlur(gray, median_size)
	
	if first_frame is None:
		first_frame = gray
		continue

	frame_diff = cv2.absdiff(first_frame,gray)
	t,thresh = cv2.threshold(frame_diff,limit_thr,255,cv2.THRESH_BINARY)
	thresh = cv2.erode(thresh,kernel,iterations=1)
	thresh = cv2.Canny(thresh,canny_thr,255)
	thresh = cv2.dilate(thresh,None,iterations=2)

	# cv2.imshow("Security Feed", frame_full)
	cv2.imshow("Thresh", thresh)
	cv2.waitKey(1)& 0xFF


print(hit)
video.release()
cv2.destroyAllWindows()
