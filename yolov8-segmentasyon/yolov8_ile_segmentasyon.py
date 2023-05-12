from ultralytics import YOLO

def main():
    model = YOLO("yolov8n-seg.pt")
    result = model.track(source=0, show=True, conf=0.5)


if __name__ == "__main__":
    main()