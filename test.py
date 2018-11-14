import cv2
import numpy as np
import sys
import imutils
import matplotlib.pyplot as plt

path = 'SourceClip.mp4'
video = cv2.VideoCapture(path)
ret,frame = video.read()
plt.imshow(frame)
plt.show()