from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load model
model = YOLO('best.pt')


# Predict and get the results
model.predict(source=0, imgsz=640, conf=0.5, show=True)
