import cv2
import numpy as np
import imutils
import skvideo
import skvideo.io
import matplotlib.pyplot as plt

src = 'SourceClip.mp4'
video = cv2.VideoCapture(src)
fps = video.get(cv2.CAP_PROP_FPS)
print(float(fps))

if not video.isOpened():
	print("Video could not be opened")
	sys.exit()


# Video writing 

# out_video = 'output.mp4'
# writer = skvideo.io.FFmpegWriter(out_video)

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out_video = 'output_vid.mp4'
# ret,f_frame = video.read()
# out_sh = f_frame.shape
# out_write= cv2.VideoWriter(out_video, fourcc, fps, (out_sh[0],out_sh[1]))

#Initializing variables for processing frames
first_frame = None
hits =0
miss =0
kernel_e = np.ones((10,10),np.uint8) # kernel for erosion
kernel_d = np.ones((5,5),np.uint8)	# kernel for dilation
kernel_b = np.ones((10,10),np.uint8) # kernel for dilation for basket
pxl_sum =0
pxl =0
frame_cnt =0
dur_cnt =0
cnt =0

# Looping through frames and detecting hit and miss
while True:

	ret,full_frame = video.read()

	if full_frame is None:
		break

	frame_cnt+=1
	frame = full_frame[230:550,450:800]
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (25,25),0)
	
	if first_frame is None:
		first_frame = gray
		continue

	frame_diff = cv2.absdiff(first_frame,gray)
	t,thresh = cv2.threshold(frame_diff,20,255,cv2.THRESH_BINARY)

	thresh = cv2.erode(thresh,kernel_e,iterations = 1)
	thresh = cv2.dilate(thresh,kernel_e, iterations=1)

	thresh = cv2.Canny(thresh,20,255)
	add = cv2.dilate(thresh[75:175,139:238], kernel_b,iterations =2)
	thresh = cv2.dilate(thresh,kernel_d, iterations=1)
	thresh[75:175,139:238] +=add


	pxl_n = len(thresh[thresh==255])
	if pxl_n >500:
		pxl_sum +=pxl_n
		cnt +=1
		pxl =1
	elif pxl_n <1:
		pxl =0
		cnt =0

	if pxl ==0: 
		if pxl_sum >40000 and (frame_cnt-dur_cnt)>100:
			hits+=1
			dur_cnt = frame_cnt
		elif pxl_sum <35000 and pxl_sum >20000 and  (frame_cnt-dur_cnt)>100:
			miss +=1
			dur_cnt =frame_cnt
		# Debug help
		# if pxl_sum !=0:    # debug help
		# 	print(pxl_sum)
		pxl_sum =0

	full_frame[0:320,1569 :1919,:] = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
	cv2.putText(full_frame, "Hit = {}".format(hits), (75,1050), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
	cv2.putText(full_frame, "Miss = {}".format(miss), (275,1050), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
	cv2.putText(full_frame, "{}".format(cnt), (1340,1050), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
	# out_write.write(full_frame)

	# Shows output frame bu frame
	cv2.imshow("Full Frame", full_frame)        
	cv2.waitKey(1)& 0xFF

video.release()
# out_write.release()
cv2.destroyAllWindows()




