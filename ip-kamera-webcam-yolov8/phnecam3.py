# yolov8 ultralytics ile nesne algılama

import cv2
from ultralytics import YOLO
#Starting the video capture
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("http://192.168.1.22:8080//video")

 
while(cap.isOpened()):
    ret,img = cap.read()
    img = cv2.resize(img, (640, 480))
    results = model(img, stream=True)

    #Controlling the algorithm with keys
    try:
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)


        cv2.imshow('img',img)
        a = cv2.waitKey(1)
        if a == ord('q'):
            break
    except cv2.error:
        print("stream ended")
        break

cap.release()
cv2.destroyAllWindows()
