import cv2
import threading
import queue
import sys
import os
import dotenv
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import subprocess

###############################################################################
# Environment Setup & YOLOv7 Model Loading
###############################################################################
dotenv.load_dotenv()

YOLOV7_PATH = os.getenv("YOLOV7_PATH")
YOLOV7_WEIGHTS = os.getenv("YOLOV7_WEIGHTS", "yolov7.pt")

if YOLOV7_PATH and YOLOV7_PATH not in sys.path:
    sys.path.append(YOLOV7_PATH)

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device

device = "cuda" if torch.cuda.is_available() else "cpu"
if not os.path.exists(YOLOV7_WEIGHTS):
    raise FileNotFoundError(f"Model weights not found: {YOLOV7_WEIGHTS}")

print("Loading YOLOv7 model...")
model = attempt_load(YOLOV7_WEIGHTS, map_location=device)
model.eval()
if device == "cuda":
    try:
        model = torch.compile(model, backend="eager")
        print("Model compiled successfully.")
    except Exception as e:
        print("Model compilation skipped due to:", e)
print(f"YOLOv7 model loaded on {device}")
if device == "cuda":
    model.half()

SPORTS_BALL_CLASS_ID = 32
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3

###############################################################################
# Overlay Setup: Assets and Helper Functions
###############################################################################
tmobile_logo = cv2.imread("assets/Tmobile_Logo.png")
seattle_logo = cv2.imread("assets/SeattleU_SponsorLogo.png")
home_logo    = cv2.imread("assets/SeattleU_Logo.png")
away_logo    = cv2.imread("assets/UW_Logo.png")

if tmobile_logo is not None:
    tmobile_logo = cv2.resize(tmobile_logo, (80, 80))
if seattle_logo is not None:
    seattle_logo = cv2.resize(seattle_logo, (80, 80))
if home_logo is not None:
    home_logo = cv2.resize(home_logo, (40, 40))
if away_logo is not None:
    away_logo = cv2.resize(away_logo, (40, 40))

FONT_PATH = "assets/FuturaMaxi.otf"
ANGLE_FONT_PATH = "assets/FuturaMaxi.otf"
FONT_SIZE = 14
ANGLE_FONT_SIZE = 12

def load_font(size, font_path):
    return ImageFont.truetype(font_path, size)

def draw_pil_text(img, text, x, y, w, h, font, text_color=(0, 0, 0)):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    centered_x = x + (w - text_w) // 2
    centered_y = y + (h - text_h) // 2 + 2
    draw.text((centered_x, centered_y), text, font=font, fill=text_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def draw_image(frame, image, x, y, width, height):
    if image is None:
        return
    frame[y:y+height, x:x+width] = image

def draw_acronym_score_box(frame, acronym, score, x, y, w, h, font):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), thickness=-1)
    mid_x = x + w // 2
    cv2.line(frame, (mid_x, y), (mid_x, y + h), (0, 0, 0), 2)
    frame = draw_pil_text(frame, acronym, x, y, w // 2, h, font, text_color=(0, 0, 0))
    frame = draw_pil_text(frame, score, mid_x, y, w // 2, h, font, text_color=(0, 0, 0))
    return frame

def draw_custom_overlay(frame, home_acronym="HOM", away_acronym="AWY",
                        home_score="0", away_score="0", action_angle=""):
    font = load_font(FONT_SIZE, FONT_PATH)
    angle_font = load_font(ANGLE_FONT_SIZE, ANGLE_FONT_PATH)
    
    scoreboard_x, scoreboard_y = 10, 10
    draw_image(frame, tmobile_logo, scoreboard_x, scoreboard_y, 80, 80)
    draw_image(frame, seattle_logo, scoreboard_x + 80, scoreboard_y, 80, 80)
    
    home_row_x = scoreboard_x + 160
    home_row_y = scoreboard_y
    draw_image(frame, home_logo, home_row_x, home_row_y, 40, 40)
    frame = draw_acronym_score_box(frame, home_acronym, home_score, home_row_x + 40, home_row_y, 80, 40, font)
    
    away_row_x = scoreboard_x + 160
    away_row_y = scoreboard_y + 40
    draw_image(frame, away_logo, away_row_x, away_row_y, 40, 40)
    frame = draw_acronym_score_box(frame, away_acronym, away_score, away_row_x + 40, away_row_y, 80, 40, font)
    
    info_box_y = away_row_y + 40
    info_box_w = 280
    info_box_h = 30
    cv2.rectangle(frame, (scoreboard_x, info_box_y), (scoreboard_x + info_box_w, info_box_y + info_box_h), (255, 255, 255), thickness=-1)
    info_text = f"CS 25.15  |  ACTION ANGLE: {action_angle}"
    frame = draw_pil_text(frame, info_text, scoreboard_x, info_box_y, info_box_w, info_box_h, angle_font, text_color=(0, 0, 0))
    return frame

###############################################################################
# Ball Detection Helper Functions (modified to also return ball coordinates)
###############################################################################
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw, dh = dw // 2, dh // 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)
    return img

def run_ball_detector(frame):
    orig_frame = frame.copy()
    img = letterbox(frame, new_shape=(640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0
    img = torch.from_numpy(img).unsqueeze(0).to(device)
    if device == "cuda":
        img = img.half()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device=="cuda")):
            pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres=CONF_THRESHOLD, iou_thres=NMS_THRESHOLD,
                               classes=[SPORTS_BALL_CLASS_ID], agnostic=False)
    ball_coords = None
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orig_frame.shape).round()
            for *xyxy, conf, cls in det:
                label = f'Soccer Ball {conf:.2f}'
                plot_one_box(xyxy, orig_frame, label=label, color=(0, 255, 0), line_thickness=2)
                x1, y1, x2, y2 = xyxy
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                ball_coords = (center_x, center_y)
                break  # use only the first detection
        if ball_coords is not None:
            break
    return orig_frame, ball_coords

###############################################################################
# Global Variables, Camera Switching Logic, and Shared Frame Storage
###############################################################################
# Define RTMP URLs for each camera.
rtmp_urls = {
    1: "rtmp://192.168.1.100/live/GoPro_SU3",
    2: "rtmp://192.168.1.100/live/GoPro_SU2",
    3: "rtmp://192.168.1.100/live/GoPro_SU1",
    4: "rtmp://192.168.1.100/live/GoPro_SU4",
}

# Create a frame queue for each camera.
frame_queues = {cam: queue.Queue(maxsize=5) for cam in rtmp_urls.keys()}

# Global state for active camera switching.
active_cam = 1  # start with camera 1
candidate_switch = None
candidate_start_time = None
SWITCH_THRESHOLD = 2.0  # seconds

# Shared dictionary to hold the latest processed frame from each camera.
latest_frames = {cam: None for cam in rtmp_urls.keys()}

def cam_switching(current_cam, ball_coords):
    """
    Determine if the active stream should be switched based on ball position.
    Splits the 1920x1080 frame into a 6x3 grid.
    """
    global candidate_switch, candidate_start_time, active_cam
    if ball_coords is None:
        candidate_switch = None
        candidate_start_time = None
        return active_cam

    x, y = ball_coords
    grid_cell_w, grid_cell_h = 320, 360  # 1920/6 and 1080/3
    col = int(x // grid_cell_w)
    row = int(y // grid_cell_h)

    # Only apply switching if the ball is in the top row.
    candidate = current_cam
    if row == 0:
        if current_cam == 1:
            candidate = 4 if col == 5 else 2
        elif current_cam == 2:
            candidate = 3 if col == 0 else 1
        elif current_cam == 3:
            candidate = 2 if col == 0 else 4
        elif current_cam == 4:
            candidate = 1 if col == 5 else 3

    # Switch only if candidate is different from current active camera.
    if candidate != active_cam:
        now = time.time()
        if candidate_switch == candidate and candidate_start_time is not None:
            if now - candidate_start_time >= SWITCH_THRESHOLD:
                print(f"Switching active stream from cam {active_cam} to cam {candidate}")
                active_cam = candidate
                candidate_switch = None
                candidate_start_time = None
        else:
            candidate_switch = candidate
            candidate_start_time = now
    else:
        candidate_switch = None
        candidate_start_time = None

    return active_cam

###############################################################################
# Asynchronous Processing for Multiple Camera Streams
###############################################################################
def frame_reader(cam_id, rtmp_url):
    cap = cv2.VideoCapture(rtmp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"Error: Unable to open stream for cam {cam_id} at {rtmp_url}")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Cam {cam_id}: frame not received, ending stream.")
            break
        if frame_queues[cam_id].full():
            continue
        frame_queues[cam_id].put(frame)
    cap.release()

def frame_processor(cam_id):
    """
    Process frames from the specific camera's queue.
    For the active stream, process every frame with full ball detection.
    For non-active streams, process every 10th frame.
    The processed frame is stored in latest_frames.
    """
    process_counter = 0
    global active_cam
    while True:
        if not frame_queues[cam_id].empty():
            frame = frame_queues[cam_id].get()
            process_counter += 1

            if cam_id == active_cam:
                proc_frame, ball_coords = run_ball_detector(frame)
                # Update active camera decision based on ball coordinates.
                current_active = cam_switching(active_cam, ball_coords)
                # Extract stream label from RTMP URL.
                stream_label = rtmp_urls[current_active].split('/')[-1]
                overlay_text = f"{stream_label}"
                final_frame = draw_custom_overlay(proc_frame, action_angle=overlay_text)
            else:
                if process_counter % 10 == 0:
                    proc_frame, _ = run_ball_detector(frame)
                    final_frame = proc_frame
                else:
                    final_frame = frame

            latest_frames[cam_id] = final_frame

        time.sleep(0.001)

###############################################################################
# Setup FFmpeg Process for Twitch Streaming
###############################################################################
# Get Twitch stream key from environment variable (set this in your .env file)
twitch_stream_key = os.getenv("TWITCH_STREAM_KEY")
if not twitch_stream_key:
    raise ValueError("TWITCH_STREAM_KEY not found in environment variables.")

twitch_url = f"rtmp://live.twitch.tv/app/{twitch_stream_key}"

# Hardcode full path to ffmpeg executable to avoid PATH lookup issues
FFMPEG_EXE = r"C:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"

# Build the FFmpeg command.
ffmpeg_cmd = [
    FFMPEG_EXE,
    "-y",  # Overwrite output if needed
    "-f", "rawvideo",  # Input is raw video data
    "-vcodec", "rawvideo",
    "-pix_fmt", "bgr24",  # OpenCV uses bgr24 format
    "-s", "1920x1080",  # Resolution of the video frames
    "-r", "60",  # Frame rate is now 60 fps
    "-i", "-",  # Read input from STDIN
    "-c:v", "libx264",  # Encode using H.264
    "-pix_fmt", "yuv420p",
    "-preset", "veryfast",
    "-f", "flv",  # Output format for streaming
    twitch_url
]

ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

###############################################################################
# Main Function: Spawning Threads, Displaying and Streaming Only the Active Stream
###############################################################################
if __name__ == "__main__":
    reader_threads = []
    processor_threads = []

    # Start reader and processor threads for each camera.
    for cam_id, url in rtmp_urls.items():
        t_reader = threading.Thread(target=frame_reader, args=(cam_id, url))
        t_processor = threading.Thread(target=frame_processor, args=(cam_id,))
        reader_threads.append(t_reader)
        processor_threads.append(t_processor)
        t_reader.start()
        t_processor.start()

    # In the main thread, display and stream only the active camera's frame.
    while True:
        active_frame = latest_frames.get(active_cam)
        if active_frame is not None:
            # Display locally in a single window.
            cv2.imshow("Active Camera Stream", active_frame)
            # Write the raw frame to FFmpeg's STDIN.
            try:
                ffmpeg_process.stdin.write(active_frame.tobytes())
            except BrokenPipeError:
                print("FFmpeg pipe closed. Exiting...")
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()

    for t in reader_threads:
        t.join()
    for t in processor_threads:
        t.join()
