from ultralytics import YOLO
import cv2

def main():
    model = YOLO("yolov8n-seg.pt")

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        print('frame_shape:', frame.shape)

        # Run YOLOv8 inference on the frame
        results = model(frame, classes=0, verbose=False)

        # if not exist person
        if results[0].masks is None:
            continue

        # get box object
        box = results[0].boxes[0].xyxy[0]
        box = box.numpy().astype(int)

        # background subtraction
        mask = (results[0].masks.data[0].numpy() * 255).astype('uint8')

        cv2.imshow('Background remove', mask)
        #cv2.imshow("Orijinal", frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break



if __name__ == "__main__":
    main()