import sys
import os
import dotenv
import cv2
import torch
import numpy as np
# Load Environment Variables First
dotenv.load_dotenv()

# Load YOLOv7 Path from .env or Use Default
YOLOV7_PATH = os.getenv("YOLOV7_PATH")  # Fallback to local directory
print(YOLOV7_PATH)


YOLOV7_WEIGHTS = os.getenv("YOLOV7_WEIGHTS", "yolov7.pt")  # Use weight path from .env

print(YOLOV7_PATH)

# Ensure YOLOv7 Path is in sys.path
if YOLOV7_PATH not in sys.path:
    sys.path.append(YOLOV7_PATH)

print(f" YOLOv7 path added: {YOLOV7_PATH}")

# YOLOv7 Imports (AFTER Setting Path)
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device

# Select Device (CUDA if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Validate YOLOv7 Weights
if not os.path.exists(YOLOV7_WEIGHTS):
    raise FileNotFoundError(f"Model weights not found: {YOLOV7_WEIGHTS}")

# Load Model
print("Loading YOLOv7 model...")
model = attempt_load(YOLOV7_WEIGHTS, map_location=device)
model.eval()
print(f"Model loaded on {device}")

# Initialize Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check if the webcam is connected.")

print("📷 Webcam streaming started. Press 'Q' to exit.")

# Video Processing Loop
while True:
    ret, frame = cap.read()
    if not ret:
        print(" Failed to grab frame")
        break

    # Prepare Frame for YOLOv7
    img = cv2.resize(frame, (640, 640))  # Optimal input size
    img = img[:, :, ::-1]
    img = np.ascontiguousarray(img) 
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0).to(device)

    # Run YOLOv7 Inference (No Gradients for Performance)
    with torch.no_grad():
        pred = model(img, augment=False)[0]

    # Apply Non-Maximum Suppression
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # Process Results
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                label = f'{int(cls)} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=(255, 0, 0), line_thickness=2)

    # Show Frame
    cv2.imshow("YOLOv7 Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("Webcam stream closed.")