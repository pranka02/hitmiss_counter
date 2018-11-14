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


first_frame = None
writer = None
hit =0
kernel = np.ones((8,8),np.uint8)
flag=1
while True:
	ret,frame_full = video.read()

	if frame_full is None:
		break

	frame = frame_full[230:550,450:800]
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	gray= cv2.GaussianBlur(gray, (15,15),0)
	if first_frame is None:
		first_frame = gray
		continue

	frame_diff = cv2.absdiff(first_frame,gray)
	t,thresh = cv2.threshold(frame_diff,5,255,cv2.THRESH_BINARY)
    #thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)
	thresh = cv2.erode(thresh,kernel,iterations=1)
	thresh = cv2.dilate(thresh,None,iterations=2)
	
        # thresh = cv2.GaussianBlur(thresh, (15,15),0)

	thresh = cv2.Canny(thresh,20,255)
	thresh = cv2.dilate(thresh,None,iterations=2)

	cv2.imshow("Security Feed", frame_full)
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
                


print(hit)
video.release()
cv2.destroyAllWindows()
