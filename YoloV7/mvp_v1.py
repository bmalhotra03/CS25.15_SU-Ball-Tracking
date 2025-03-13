import cv2
import threading
import queue
import sys
import os
import dotenv
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

###############################################################################
# Environment Setup & YOLOv7 Model Loading
###############################################################################
dotenv.load_dotenv()

# Set YOLO paths from environment variables
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

# Select device (CUDA or CPU) and load model
device = "cuda" if torch.cuda.is_available() else "cpu"
if not os.path.exists(YOLOV7_WEIGHTS):
    raise FileNotFoundError(f"Model weights not found: {YOLOV7_WEIGHTS}")

print("Loading YOLOv7 model...")
model = attempt_load(YOLOV7_WEIGHTS, map_location=device)
model.eval()

# (Optional) Compile the model for CUDA
if device == "cuda":
    try:
        model = torch.compile(model, backend="eager")
        print("Model compiled successfully.")
    except Exception as e:
        print("Model compilation skipped due to:", e)
print(f"YOLOv7 model loaded on {device}")

# Convert model to FP16 if using CUDA
if device == "cuda":
    model.half()

# Detection parameters (soccer ball class from COCO)
SPORTS_BALL_CLASS_ID = 32
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3

###############################################################################
# Overlay Setup: Assets and Helper Functions
###############################################################################
# Load overlay logos (ensure the assets exist at these paths)
tmobile_logo = cv2.imread("assets/Tmobile_Logo.png")
seattle_logo = cv2.imread("assets/SeattleU_SponsorLogo.png")
home_logo    = cv2.imread("assets/SeattleU_Logo.png")
away_logo    = cv2.imread("assets/UW_Logo.png")

# Pre-resize logos to avoid resizing every frame
if tmobile_logo is not None:
    tmobile_logo = cv2.resize(tmobile_logo, (80, 80))
if seattle_logo is not None:
    seattle_logo = cv2.resize(seattle_logo, (80, 80))
if home_logo is not None:
    home_logo = cv2.resize(home_logo, (40, 40))
if away_logo is not None:
    away_logo = cv2.resize(away_logo, (40, 40))

# Custom font settings
FONT_PATH = "assets/FuturaMaxi.otf"
ANGLE_FONT_PATH = "assets/FuturaMaxi.otf"
FONT_SIZE = 16
ANGLE_FONT_SIZE = 12

def load_font(size, font_path):
    return ImageFont.truetype(font_path, size)

def draw_pil_text(img, text, x, y, w, h, font, text_color=(0, 0, 0)):
    """
    Draw text centered in a box using PIL for improved vertical alignment.
    """
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    centered_x = x + (w - text_w) // 2
    centered_y = y + (h - text_h) // 2 + 2  # slight vertical adjustment
    draw.text((centered_x, centered_y), text, font=font, fill=text_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def draw_image(frame, image, x, y, width, height):
    """
    Draw a pre-resized image (logo) on the frame.
    """
    if image is None:
        return
    frame[y:y+height, x:x+width] = image

def draw_acronym_score_box(frame, acronym, score, x, y, w, h, font):
    """
    Draw a white box with a vertical black separator and centered text.
    """
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), thickness=-1)
    mid_x = x + w // 2
    cv2.line(frame, (mid_x, y), (mid_x, y + h), (0, 0, 0), 2)
    frame = draw_pil_text(frame, acronym, x, y, w // 2, h, font, text_color=(0, 0, 0))
    frame = draw_pil_text(frame, score, mid_x, y, w // 2, h, font, text_color=(0, 0, 0))
    return frame

def draw_custom_overlay(frame, home_acronym="HOM", away_acronym="AWY",
                        home_score="0", away_score="0", action_angle=""):
    """
    Draws the overlay with logos, team score boxes, and an information row.
    """
    font = load_font(FONT_SIZE, FONT_PATH)
    angle_font = load_font(ANGLE_FONT_SIZE, ANGLE_FONT_PATH)
    
    # 1) Draw T-Mobile and Sponsor logos
    scoreboard_x, scoreboard_y = 10, 10
    draw_image(frame, tmobile_logo, scoreboard_x, scoreboard_y, 80, 80)
    draw_image(frame, seattle_logo, scoreboard_x + 80, scoreboard_y, 80, 80)
    
    # 2) Home row: team logo and score box
    home_row_x = scoreboard_x + 160
    home_row_y = scoreboard_y
    draw_image(frame, home_logo, home_row_x, home_row_y, 40, 40)
    frame = draw_acronym_score_box(frame, home_acronym, home_score, home_row_x + 40, home_row_y, 80, 40, font)
    
    # 3) Away row: team logo and score box
    away_row_x = scoreboard_x + 160
    away_row_y = scoreboard_y + 40
    draw_image(frame, away_logo, away_row_x, away_row_y, 40, 40)
    frame = draw_acronym_score_box(frame, away_acronym, away_score, away_row_x + 40, away_row_y, 80, 40, font)
    
    # 4) Info row: displays action angle (e.g., camera source)
    info_box_y = away_row_y + 40
    info_box_w = 280
    info_box_h = 30
    cv2.rectangle(frame, (scoreboard_x, info_box_y), (scoreboard_x + info_box_w, info_box_y + info_box_h), (255, 255, 255), thickness=-1)
    info_text = f"CS 25.15  |  ACTION ANGLE: {action_angle}"
    frame = draw_pil_text(frame, info_text, scoreboard_x, info_box_y, info_box_w, info_box_h, angle_font, text_color=(0, 0, 0))
    return frame

###############################################################################
# Ball Detection Helper Functions
###############################################################################
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resizes an image with unchanged aspect ratio using padding.
    """
    shape = img.shape[:2]  # current height and width
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw, dh = dw // 2, dh // 2  # evenly distribute padding
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)
    return img

def run_ball_detector(frame):
    """
    Uses YOLOv7 to detect the soccer ball on the frame and draws bounding boxes.
    """
    orig_frame = frame.copy()
    img = letterbox(frame, new_shape=(640, 640))
    # Convert BGR to RGB and HWC to CHW
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0  # Normalize
    img = torch.from_numpy(img).unsqueeze(0).to(device)
    
    # Convert input to FP16 if using CUDA
    if device == "cuda":
        img = img.half()
    
    with torch.no_grad():
        # Use automatic mixed precision for faster inference
        with torch.cuda.amp.autocast(enabled=(device=="cuda")):
            pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres=CONF_THRESHOLD, iou_thres=NMS_THRESHOLD,
                               classes=[SPORTS_BALL_CLASS_ID], agnostic=False)
    
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orig_frame.shape).round()
            for *xyxy, conf, cls in det:
                label = f'Soccer Ball {conf:.2f}'
                plot_one_box(xyxy, orig_frame, label=label, color=(0, 255, 0), line_thickness=2)
    return orig_frame

###############################################################################
# Asynchronous Processing with Queue and Threads
###############################################################################
frame_queue = queue.Queue(maxsize=5)  # small buffer to minimize lag

def frame_reader(rtmp_url):
    """
    Continuously reads frames from the RTMP stream and pushes them to a queue.
    """
    cap = cv2.VideoCapture(rtmp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"Error: Unable to open stream {rtmp_url}")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Frame not received, ending stream.")
            break
        # If the queue is full, skip the frame to avoid delay
        if frame_queue.full():
            continue
        frame_queue.put(frame)
    cap.release()

def frame_processor(window_name, action_label="Unknown"):
    """
    Processes frames from the queue, runs inference, overlays graphics, and displays the result.
    """
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            processed_frame = run_ball_detector(frame)
            final_frame = draw_custom_overlay(processed_frame,
                                              home_acronym="SU", away_acronym="UW",
                                              home_score="0", away_score="0",
                                              action_angle=action_label)
            cv2.imshow(window_name, final_frame)
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow(window_name)

###############################################################################
# Main Function: Start Streaming Threads
###############################################################################
if __name__ == "__main__":
    # Define your RTMP stream URL (adjust as needed)
    rtmp_url = "rtmp://192.168.1.100/live/GoPro_SU3"
    action_label = rtmp_url.split('/')[-1]

    # Start the reader and processor threads
    reader_thread = threading.Thread(target=frame_reader, args=(rtmp_url,))
    processor_thread = threading.Thread(target=frame_processor, args=("GoPro Stream", action_label))
    reader_thread.start()
    processor_thread.start()
    reader_thread.join()
    processor_thread.join()
    cv2.destroyAllWindows()