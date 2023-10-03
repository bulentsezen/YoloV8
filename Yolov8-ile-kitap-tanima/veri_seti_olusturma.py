from ultralytics import YOLO
import cv2
import cvzone
import math
from time import time

#############################
classID = 1  # 0 Canakkale, 1 Tesla
save = True
outputFolderPath = 'Dataset/DataCollect'
blurThreshold = 35  # Larger is more focus
floatingPoint = 6
camWidth, camHeight = 640, 480
debug = True

#############################

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

cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, camWidth)
cap.set(4, camHeight)

model = YOLO("yolov8n.pt")

while True:
    success, img = cap.read()
    imgOut = img.copy()
    results = model(img, stream=True)

    listBlur = []  # True False values indicating if the faces are blur or not
    listInfo = []  # The normalized values and the class name for the label txt file

    for r in results:
        boxes = r.boxes
        for box in boxes:
            if int(box.cls[0]) == 73:
                # print(int(box.cls[0]))
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                w, h = x2 - x1, y2 - y1

                cvzone.cornerRect(img, (x1, y1, w, h))
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100

                # ------  Find Blurriness --------
                imgFace = img[y1:y1 + h, x1:x1 + w]
                # cv2.imshow("Face", imgFace)
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                if blurValue > blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)

                # ------  Normalize Values  --------
                ih, iw, _ = img.shape
                xc, yc = x1 + w / 2, y1 + h / 2

                xcn, ycn = round(xc / iw, floatingPoint), round(yc / ih, floatingPoint)
                wn, hn = round(w / iw, floatingPoint), round(h / ih, floatingPoint)
                # print(xcn, ycn, wn, hn)

                # ------  To avoid values above 1 --------
                if xcn > 1: xcn = 1
                if ycn > 1: ycn = 1
                if wn > 1: wn = 1
                if hn > 1: hn = 1

                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                # Class Name
                cls = int(box.cls[0])

                cvzone.putTextRect(img, f'{classNames[cls]} {conf} {blurValue}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

                if debug:
                    cv2.rectangle(img, (x1, y1, w, h), (255, 0, 0), 3)
                    cvzone.putTextRect(img, f'% Blur: {blurValue}', (x1, y1 - 0),
                                       scale=2, thickness=3)

                # ------  To Save --------
                if save:
                    # ------  Save Image  --------
                    timeNow = time()
                    timeNow = str(timeNow).split('.')
                    timeNow = timeNow[0] + timeNow[1]
                    cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", imgOut)
                    # ------  Save Label Text File  --------
                    for info in listInfo:
                        f = open(f"{outputFolderPath}/{timeNow}.txt", 'a')
                        f.write(info)
                        f.close()

    cv2.imshow("Image", img)
    cv2.waitKey(1)
