from ultralytics import YOLO
import cv2
import numpy as np

def main():
    model = YOLO("yolov8n-seg.pt")   # önceden eğitilmiş ağırlıklar
    model.classes = ['cell phone']

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        print('frame_shape:', frame.shape)

        # Run YOLOv8 inference on the frame
        results = model(frame, verbose=False)    # classes=0 silindi

        # if not exist person
        if results[0].masks is None:
            continue

        # get box object
        box = results[0].boxes[0].xyxy[0]
        box = box.numpy().astype(int)

        # background subtraction
        mask = (results[0].masks.data[0].numpy() * 255).astype('uint8')

        beyaz_pixel_sayisi = np.sum(mask == 255)  # sadece beyaz pixelleri sayar
        sihay_pixel_sayisi = np.sum(mask == 0)  # sadece siyah pixelleri sayar
        toplam_pixel_sayisi = beyaz_pixel_sayisi + sihay_pixel_sayisi
        beyaz_yuzde = int(beyaz_pixel_sayisi / toplam_pixel_sayisi * 100)
        print(beyaz_yuzde)

        cv2.putText(mask, f'Doluluk: %{str(beyaz_yuzde)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 0), 3)

        cv2.imshow('Background remove', mask)
        cv2.imshow("Orijinal", frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break



if __name__ == "__main__":
    main()