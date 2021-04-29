import cv2
import numpy as np

def nothing(x):
    pass

frame = cv2.imread("osa1.jpg")
scale_percent = 10 # percent of original size
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height) 
frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 0, 255, nothing)
while True:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("result", result)
    key = cv2.waitKey(1)
    if key == 27:
        break
