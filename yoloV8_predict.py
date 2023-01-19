from ultralytics import YOLO
from PIL import Image

# modeli yükle
model = YOLO('yolov8n.pt')  # load a pretrained model

# modeli kullanarak video ve webcam görüntüsü ile nesne tahmini yap
#sonuc = model.predict(source="0")     # 0 webcam için
#sonuc = model.predict(source="Video.mp4", show=True)

# resim dosyası üzerinde nesne tanıma
im1 = Image.open("Elon_Musk.jpg")
sonuc = model.predict(source=im1, save=True)  # save ile resmi kaydeder