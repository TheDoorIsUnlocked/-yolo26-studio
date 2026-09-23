from ultralytics import YOLO

# Load model
model = YOLO("yolo26n.pt")
# Train the model
model.train(data="icon.yaml")
# model.train(resume=True) 接着上次训练继续训练
