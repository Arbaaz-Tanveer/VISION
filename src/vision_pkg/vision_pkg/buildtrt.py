from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("yolo11n.pt")

# Export the model to TensorRT
model.export(format="engine",batch = 4,half = True)  # creates 'yolo11n.engine'
