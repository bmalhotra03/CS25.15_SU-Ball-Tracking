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
print(f"YOLOv7 model loaded on {device}")

# List of RTMP stream URLs
rtmp_urls = [
    #"rtmp://192.168.1.100/live/GoPro_SU1",
    #"rtmp://192.168.1.100/live/GoPro_SU2",
    "rtmp://192.168.1.100/live/GoPro_SU3",
    #"rtmp://192.168.1.100/live/GoPro_SU4"
]

# (Optional) Create a lock if you experience threading issues with the model
model_lock = threading.Lock()

def run_yolo(frame):
    """
    Process a frame using YOLOv7:
      - Preprocess the frame (resize, convert color, tensor conversion)
      - Run inference with YOLO
      - Draw bounding boxes on the original frame using plot_one_box
    """
    orig_frame = frame.copy()
    # Preprocess: resize to YOLO's input size (640, 640)
    img = cv2.resize(frame, (640, 640))
    img = img[:, :, ::-1]  # BGR to RGB
    img = np.ascontiguousarray(img)
    img = img.transpose(2, 0, 1)  # Change to CHW format
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0).to(device)

    # Run inference (with optional locking if needed)
    with torch.no_grad():
        with model_lock:
            pred = model(img, augment=False)[0]

    # Apply non-max suppression
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # Process detections and draw boxes
    for det in pred:
        if det is not None and len(det):
            # Rescale coordinates to original frame size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orig_frame.shape).round()
            for *xyxy, conf, cls in det:
                label = f'{int(cls)} {conf:.2f}'
                plot_one_box(xyxy, orig_frame, label=label, color=(255, 0, 0), line_thickness=2)
    return orig_frame

def stream_video(rtmp_url, window_name):
    cap = cv2.VideoCapture(rtmp_url)
    if not cap.isOpened():
        print(f"Error: Unable to open the RTMP stream for {window_name}")
        return

    # Set resolution to Full HD
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print(f"Streaming from {window_name} with YOLO detection... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Unable to grab frame from {window_name}")
            break

        # Process frame using YOLO for object detection
        processed_frame = run_yolo(frame)

        # Display the processed frame
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
