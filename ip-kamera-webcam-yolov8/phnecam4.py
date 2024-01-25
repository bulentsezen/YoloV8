# sadece insan yada sadece araba algılatma

import cv2
from ultralytics import YOLO
#Starting the video capture
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("http://192.168.1.22:8080//video")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

 
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


                # Class Name
                cls = int(box.cls[0])
                print(cls)

                # 0 insan 2 araba
                if cls == 0:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)


        cv2.imshow('img',img)
        a = cv2.waitKey(1)
        if a == ord('q'):
            break
    except cv2.error:
        print("stream ended")
        break

cap.release()
cv2.destroyAllWindows()
