from ultralytics import YOLO

model = YOLO('yolov8n.pt')

sonuc = model.track(source=0, show=True, tracker="bytetrack.yaml")
# sonuc = model.track(source="cctv_trafik.mp4", show=True, tracker="bytetrack.yaml")
