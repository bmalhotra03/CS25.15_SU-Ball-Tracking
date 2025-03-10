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

# Define the class ID for a sports ball (soccer ball)
SPORTS_BALL_CLASS_ID = 67
# Input size for ball detection as used in ball_tracking
INPUT_SIZE = 320

# (Optional) Create a lock if you experience threading issues with the model
model_lock = threading.Lock()

def run_ball_detector(frame):
    """
    Process a frame using YOLOv7 to detect ONLY a soccer ball.
    Preprocesses the frame to the expected size (320x320), runs inference with
    a class filter (SPORTS_BALL_CLASS_ID) and draws bounding boxes on the original frame.
    """
    orig_frame = frame.copy()
    # Resize frame to the input size used in ball_tracking
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    # Convert BGR to RGB and change data layout from HWC to CHW
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = img.astype('float32') / 255.0  # Normalize
    img = torch.from_numpy(img).unsqueeze(0).to(device)

    # Run inference with torch.autocast if available for speed
    with torch.no_grad():
        with torch.autocast("cuda", enabled=torch.cuda.is_available()):
            with model_lock:
                pred = model(img, augment=False)[0]
    # Apply non-max suppression limiting detections to the sports ball class
    pred = non_max_suppression(pred, conf_thres=0.1, iou_thres=0.45, classes=[SPORTS_BALL_CLASS_ID], agnostic=False)

    # Process detections and draw boxes on the original frame
    for det in pred:
        if det is not None and len(det):
            # Rescale coordinates to the original frame size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orig_frame.shape).round()
            for *xyxy, conf, cls in det:
                label = f'soccer ball {conf:.2f}'
                # Draw bounding box with a distinct color (green)
                plot_one_box(xyxy, orig_frame, label=label, color=(0, 255, 0), line_thickness=2)
    return orig_frame

# List of RTMP stream URLs
rtmp_urls = [
    "rtmp://192.168.1.100/live/GoPro_SU1",
    "rtmp://192.168.1.100/live/GoPro_SU2",
    "rtmp://192.168.1.100/live/GoPro_SU3",
    "rtmp://192.168.1.100/live/GoPro_SU4"
]

def stream_video(rtmp_url, window_name):
    cap = cv2.VideoCapture(rtmp_url)
    if not cap.isOpened():
        print(f"Error: Unable to open the RTMP stream for {window_name}")
        return

    # Set resolution to Full HD
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    print(f"Streaming from {window_name} with soccer ball detection... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Unable to grab frame from {window_name}")
            break

        # Process frame using the soccer ball–only detector
        processed_frame = run_ball_detector(frame)

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