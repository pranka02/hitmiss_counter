import cv2
import numpy as np
import sys
import imutils
import time
import matplotlib.pyplot as plt

template = cv2.imread('basket.jepg',0)
hits = 0
miss =0

path = 'SourceClip.mp4'
video = cv2.VideoCapture(path)
ret,full_frame = video.read()

frame = full_frame[230:430,450:800]
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

full_frame[0:200,1569 :1919,:] = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
cv2.putText(full_frame, "Hit: {}".format(hits), (75,1050), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
cv2.putText(full_frame, "Miss: {}".format(miss), (275,1050), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
# cv2.putText(full_frame, "Miss: {}".format(miss), (275,1050), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
# cv2.imshow("gray",frame)
# cv2.waitKey()

plt.imshow(full_frame)
plt.show()


# img1 = template     #cv2.Canny(template,0,255)        # queryImage
# img2 = gray				#cv2.Canny(gray,0,255)  # trainImage

# # Initiate SIFT detector
# orb = cv2.ORB_create()

# # find the keypoints and descriptors with SIFT
# kp1, des1 = orb.detectAndCompute(img1,None)
# kp2, des2 = orb.detectAndCompute(img2,None)


# # create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# # Match descriptors.
# matches = bf.match(des1,des2)

# # Sort them in the order of their distance.
# matches = sorted(matches, key = lambda x:x.distance)

# # Draw first 10 matches.
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],outImg =img1)

# plt.imshow(img3),plt.show()

# print(matches)

# # Store width and heigth of template in w and h 
# w, h = template.shape[::-1] 
  
# # Perform match operations. 
# res = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED) 
  
# # Specify a threshold 
# threshold = 0.5
  
# # Store the coordinates of matched area in a numpy array 
# loc = np.where( res >= threshold)  
  
# # Draw a rectangle around the matched region. 
# for pt in zip(*loc[::-1]): 
#     cv2.rectangle(gray, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2) 
  
# # Show the final image with the matched area. 
# cv2.imshow('Detected',gray) 

# cv2.imshow("gray",template)
# cv2.waitKey()

# print(loc)
# then = time.time()
# kernel_d = np.ones((3,3),np.uint8)
# frame = full_frame[230:500,450:800]
# gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
# now = then+1
# sec = now/60
# milsec = 
# print(time.time_ns())





























# edges =cv2.Canny(gray,50,100)
# thresh = cv2.dilate(edges,kernel_d,iterations=1)
# thresh = cv2.erode(thresh,kernel_d,iterations=1)



# params = cv2.SimpleBlobDetector_Params()

# # Change thresholds
# params.minThreshold = 20
# params.maxThreshold = 255


# # Filter by Area.
# params.filterByArea = True
# params.minArea = 2500

# # Filter by Circularity
# params.filterByCircularity = True
# params.minCircularity = 0.4

# # Filter by Convexity
# params.filterByConvexity = True
# params.minConvexity = 0.87

# # Filter by Inertia
# params.filterByInertia = True
# params.minInertiaRatio = 0.1

# detector = cv2.SimpleBlobDetector_create()
# keypoints = detector.detect(thresh)
# im_with_keypoints = cv2.drawKeypoints(thresh, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# edges =cv2.Canny(gray,50,100)
# thresh = cv2.dilate(edges,kernel_d,iterations=1)
# thresh = cv2.erode(thresh,kernel_d,iterations=1)
# thresh = cv2.dilate(thresh,kernel_d,iterations=2)
# plt.imshow(frame)
# plt.show()
# ga = np.array(gray)
# gray[gray>180] =0
