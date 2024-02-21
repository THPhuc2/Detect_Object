import cv2
import numpy as np


df = "D:\CODE\DEEP_LEARNING__COMPUTER_VISION\Computer_vision\Project\Detect_object\data"
# img = cv2.imread(df + "\\messi.jpg")
cap = cv2.VideoCapture(df + "\vtest.mp4")
# fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows= False)
# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
fgbg = cv2.createBackgroundSubtractorKNN(detectShadows= True)
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
while cap.isOpened():
    ret, frame = cap.read()

    if frame is None:
        break

    fgmask = fgbg.apply(frame)
    # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    cv2.imshow("frame", frame)

    cv2.imshow("FG MASK frame", fgmask)

    keyboard = cv2.waitKey(30)

    if keyboard == "q" or keyboard == 27:
        break

cap.release()
cv2.destroyAllWindows()