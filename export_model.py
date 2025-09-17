from ultralytics import YOLO

# Cargar el modelo nano 
model = YOLO("yolov8n.pt")

# Exportar a ONNX
model.export(format="onnx", opset=12)