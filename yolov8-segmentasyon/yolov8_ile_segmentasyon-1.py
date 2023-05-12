from ultralytics import YOLO
import cv2

def main():
    model = YOLO("yolov8n-seg.pt")

    for result in model.track(source=0, show=True, stream=True):
        frame = result.orig_img

        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):
            break

if __name__ == "__main__":
    main()