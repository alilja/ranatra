import numpy as np
import cv2

cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=100000)

for i in range(10):
    init_frame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)

while(1):
    rep, frame = cap.read()
    fgmask = fgbg.apply(frame)

    #frame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
    #fgmask = init_frame - frame

    blurred = cv2.GaussianBlur(fgmask, (45, 45), 0)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('Thresholded', thresh)
    #cv2.imshow('frame', fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()