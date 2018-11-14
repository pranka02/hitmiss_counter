import cv2
import numpy as np
import sys
import imutils
import datetime
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


first_frame = None
writer = None
hits =0
miss =0
kernel = np.ones((8,8),np.uint8)
flag=1
pxl_sum =0
pxl =0
while True:
	ret,frame_full = video.read()

	if frame_full is None:
		break

	frame = frame_full[230:550,450:800]
	frame_full = imutils.resize(frame_full,width=700,height=700)
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	gray= cv2.GaussianBlur(gray, (15,15),0)

	if first_frame is None:
		first_frame = gray
		continue

	frame_diff = cv2.absdiff(first_frame,gray)
	t,thresh = cv2.threshold(frame_diff,20,255,cv2.THRESH_BINARY)
  
	thresh = cv2.erode(thresh,kernel,iterations=1)
	thresh = cv2.dilate(thresh,None,iterations=2)

	thresh = cv2.Canny(thresh,20,255)
	thresh = cv2.dilate(thresh,None,iterations=1)

	pxl_n = len(thresh[thresh==255])
	if pxl_n >800:
		pxl_sum +=pxl_n
		pxl =1
	elif pxl_n <1:
		pxl =0
	# print(pxl_n)
	if pxl ==0: 
		if pxl_sum >10000:
			hits+=1
		elif pxl_sum <10000 and pxl_sum >3000:
			miss +=1
		if pxl_sum !=0:
			print(pxl_sum)
		pxl_sum =0



	cv2.putText(frame_full, "Hits: {}".format(hits), (20,380), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
	cv2.putText(frame_full, "Miss: {}".format(miss), (90,380), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
	
	cv2.imshow("Full Frame", frame_full)
	cv2.imshow("Thresh", thresh)

	if flag is 0:
		k=cv2.waitKey(0)& 0xFF #move with keys
		if k is 27:
			destroyAllWindows()
		elif k is ord('p'):
			flag=1
	elif flag is 1:
		k=cv2.waitKey(1)& 0xFF #play
		if k is 27:
			destroyAllWindows()
		elif k is ord('p'):
			flag=0
                


print(hits)
video.release()
cv2.destroyAllWindows()
