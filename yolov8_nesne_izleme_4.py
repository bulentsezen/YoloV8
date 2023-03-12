from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from sort import *

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

mask = cv2.imread("maske.jpg")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

prev_frame_time = 0
new_frame_time = 0

# cap = cv2.VideoCapture(1)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)

cap = cv2.VideoCapture("arabalar.mp4")

# video kayıt için fourcc ve VideoWriter tanımlama
cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
success, img = cap.read()
# print(img.shape)
# cv2.imwrite("ornek_resim.jpg", img)
size = list(img.shape)
del size[2]
size.reverse()
video = cv2.VideoWriter("kaydedilen_video.mp4", cv2_fourcc, 24, size) #output video name, fourcc, fps, size

model = YOLO("yolov8n.pt")

while True:
    new_frame_time = time.time()
    success, img = cap.read()

    imgRegion = cv2.bitwise_and(img, mask)

    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

            # cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

            currentArray = np.array([x1, y1, x2, y2, conf])
            detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

    # video kayıt
    video.write(img)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print("fps: ", fps)



    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)

video.release()