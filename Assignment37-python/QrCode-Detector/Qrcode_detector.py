import cv2
from pyzbar.pyzbar import decode
import numpy as np


video = cv2.VideoCapture(0)


while True:
    ret,frame = video.read()
    
    if ret == False:
        break

    gray_img = cv2.cvtColor(frame,0)
    barcode = decode(gray_img)

    for obj in barcode:
        points = obj.polygon
        print(points)
        (x,y,w,h) = obj.rect
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        result = cv2.drawContours(frame,[pts],-1,(0,255,0),4)
        barcodeData = obj.data.decode("utf-8")
        barcodeType = obj.type
        string = "Data " + str(barcodeData) + " | Type " + str(barcodeType)
        
        cv2.putText(frame, string, (x,y), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0), 2)

    cv2.imshow('Image', frame)
   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.waitKey(10)

    