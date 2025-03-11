import cv2
import threading
import sys
import os
import dotenv
import torch
import numpy as np

# Load environment variables (e.g. YOLOV7_PATH, YOLOV7_WEIGHTS)
dotenv.load_dotenv()

# Set YOLO paths from the environment
YOLOV7_PATH = os.getenv("YOLOV7_PATH")
YOLOV7_WEIGHTS = os.getenv("YOLOV7_WEIGHTS", "yolov7.pt")

# Add YOLOv7 path to sys.path if needed
if YOLOV7_PATH and YOLOV7_PATH not in sys.path:
    sys.path.append(YOLOV7_PATH)

# Import YOLOv7 modules
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device

# Set device and load YOLOv7 model
device = "cuda" if torch.cuda.is_available() else "cpu"
if not os.path.exists(YOLOV7_WEIGHTS):
    raise FileNotFoundError(f"Model weights not found: {YOLOV7_WEIGHTS}")

print("Loading YOLOv7 model...")
model = attempt_load(YOLOV7_WEIGHTS, map_location=device)
model.eval()

# (Optional) Compile the model for performance if using CUDA
if device == "cuda":
    try:
        model = torch.compile(model, backend="eager")
        print("Model compiled successfully.")
    except Exception as e:
        print("Model compilation skipped due to:", e)

print(f"YOLOv7 model loaded on {device}")

# Define the class ID for a soccer ball (correct COCO dataset class)
SPORTS_BALL_CLASS_ID = 32  # Correct class ID
# Adjust confidence and NMS threshold for better accuracy
CONF_THRESHOLD = 0.5  # Increased for better accuracy
NMS_THRESHOLD = 0.3  # Lowered to prevent oversized bounding boxes

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resizes image while maintaining aspect ratio using padding.
    """
    shape = img.shape[:2]  # current shape [height, width]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # padding
    dw, dh = dw // 2, dh // 2  # divide padding evenly
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)  # add border
    return img

def run_ball_detector(frame):
    """
    Process a frame using YOLOv7 to detect ONLY a soccer ball with better accuracy.
    """
    orig_frame = frame.copy()
    # Resize frame while maintaining aspect ratio using custom letterbox function
    img = letterbox(frame, new_shape=(640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # Convert BGR to RGB and change format
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0  # Normalize
    img = torch.from_numpy(img).unsqueeze(0).to(device)

    # Run inference with torch.autocast for better performance
    with torch.no_grad():
        with torch.autocast("cuda", enabled=torch.cuda.is_available()):
            pred = model(img, augment=False)[0]
    # Apply non-max suppression, limiting detections to the sports ball class
    pred = non_max_suppression(pred, conf_thres=CONF_THRESHOLD, iou_thres=NMS_THRESHOLD, classes=[SPORTS_BALL_CLASS_ID], agnostic=False)

    # Process detections and draw tighter bounding boxes on the original frame
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orig_frame.shape).round()
            for *xyxy, conf, cls in det:
                label = f'Soccer Ball {conf:.2f}'
                # Draw bounding box with improved accuracy
                plot_one_box(xyxy, orig_frame, label=label, color=(0, 255, 0), line_thickness=2)
    return orig_frame

# List of RTMP stream URLs
rtmp_urls = [
    "rtmp://192.168.1.100/live/GoPro_SU3",
]

def stream_video(rtmp_url, window_name):
    cap = cv2.VideoCapture(rtmp_url)
    if not cap.isOpened():
        print(f"Error: Unable to open the RTMP stream for {window_name}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print(f"Streaming from {window_name} with refined soccer ball detection... Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Unable to grab frame from {window_name}")
            break

        processed_frame = run_ball_detector(frame)
        cv2.imshow(window_name, processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(window_name)

# Start a thread for each RTMP stream
threads = []
for i, url in enumerate(rtmp_urls):
    thread = threading.Thread(target=stream_video, args=(url, f"GoPro {i+1}"))
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()
cv2.destroyAllWindows()