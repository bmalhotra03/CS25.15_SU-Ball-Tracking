import os
import sys
import dotenv
# 1) Load environment variables (including YOLOV7_PATH)
dotenv.load_dotenv()

# 2) Prepend YOLOv7 repo path to sys.path so we can import utils/ and models/
YOLOV7_PATH = os.getenv("YOLOV7_PATH")
if not YOLOV7_PATH:
    raise RuntimeError("YOLOV7_PATH not set in .env")
if YOLOV7_PATH not in sys.path:
    sys.path.insert(0, YOLOV7_PATH)

import cv2
import threading
import queue
import sys  # noqa: F401
import os   # noqa: F401
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import subprocess

# now safe to import YOLOv7 internals
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device

###############################################################################
# Environment Setup & YOLOv7 Model Loading
###############################################################################
YOLOV7_WEIGHTS = os.getenv("YOLOV7_WEIGHTS", "yolov7.pt")

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
    model.half()
print(f"YOLOv7 model loaded on {device}")

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

# safe checks on imread results
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
FONT_SIZE = 12
ANGLE_FONT_SIZE = 12

def load_font(size, font_path):
    return ImageFont.truetype(font_path, size)

def draw_pil_text(img, text, x, y, w, h, font, text_color=(0,0,0)):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    bbox = draw.textbbox((0,0), text, font=font)
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
    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255), thickness=-1)
    mid_x = x + w//2
    cv2.line(frame, (mid_x, y), (mid_x, y+h), (0,0,0), 2)
    frame = draw_pil_text(frame, acronym, x, y, w//2, h, font, text_color=(0,0,0))
    frame = draw_pil_text(frame, score, mid_x, y, w//2, h, font, text_color=(0,0,0))
    return frame

def draw_custom_overlay(frame, home_acronym="HOM", away_acronym="AWY",
                        home_score="0", away_score="0", action_angle=""):
    font = load_font(FONT_SIZE, FONT_PATH)
    angle_font = load_font(ANGLE_FONT_SIZE, ANGLE_FONT_PATH)

    scoreboard_x, scoreboard_y = 10, 10
    draw_image(frame, tmobile_logo, scoreboard_x, scoreboard_y, 80, 80)
    draw_image(frame, seattle_logo, scoreboard_x+80, scoreboard_y, 80, 80)

    home_row_x = scoreboard_x + 160
    home_row_y = scoreboard_y
    draw_image(frame, home_logo, home_row_x, home_row_y, 40, 40)
    frame = draw_acronym_score_box(frame, home_acronym, home_score,
                                   home_row_x+40, home_row_y, 80, 40, font)

    away_row_x = scoreboard_x + 160
    away_row_y = scoreboard_y + 40
    draw_image(frame, away_logo, away_row_x, away_row_y, 40, 40)
    frame = draw_acronym_score_box(frame, away_acronym, away_score,
                                   away_row_x+40, away_row_y, 80, 40, font)

    info_box_y = away_row_y + 40
    info_box_w = 280
    info_box_h = 30
    cv2.rectangle(frame,
                  (scoreboard_x, info_box_y),
                  (scoreboard_x+info_box_w, info_box_y+info_box_h),
                  (255,255,255), thickness=-1)
    info_text = f"CS 25.15  |  ACTION ANGLE: {action_angle}"
    frame = draw_pil_text(frame, info_text,
                          scoreboard_x, info_box_y,
                          info_box_w, info_box_h,
                          angle_font, text_color=(0,0,0))
    return frame

###############################################################################
# Ball Detection Helper Functions
###############################################################################
def letterbox(img, new_shape=(640,640), color=(114,114,114)):
    shape = img.shape[:2]
    ratio = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
    new_unpad = (int(round(shape[1]*ratio)), int(round(shape[0]*ratio)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw, dh = dw//2, dh//2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, dh, dh, dw, dw,
                             cv2.BORDER_CONSTANT, value=color)
    return img

def run_ball_detector(frame):
    orig = frame.copy()
    img = letterbox(frame, new_shape=(640,640))
    img = img[:,:,::-1].transpose(2,0,1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0
    img = torch.from_numpy(img).unsqueeze(0).to(device)
    if device=="cuda":
        img = img.half()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device=="cuda")):
            pred = model(img, augment=False)[0]
    det = non_max_suppression(pred,
                              conf_thres=CONF_THRESHOLD,
                              iou_thres=NMS_THRESHOLD,
                              classes=[SPORTS_BALL_CLASS_ID])[0]
    ball_coords = None
    if det is not None and len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orig.shape).round()
        x1,y1,x2,y2,conf,_ = det[0]
        plot_one_box([x1,y1,x2,y2], orig,
                     label=f"{conf:.2f}",
                     color=(0,255,0), line_thickness=2)
        ball_coords = ((x1+x2)/2, (y1+y2)/2)
    return orig, ball_coords

###############################################################################
# Global Variables, Camera Switching Logic, and Shared Frame Storage
###############################################################################
rtmp_urls = {
    1: "rtmp://192.168.1.100/live/GoPro_SU3",
    #2: "rtmp://192.168.1.100/live/GoPro_SU2",
    #3: "rtmp://192.168.1.100/live/GoPro_SU1",
    #4: "rtmp://192.168.1.100/live/GoPro_SU4",
}
frame_queues = {cam: queue.Queue(maxsize=5) for cam in rtmp_urls}
latest_frames = {cam: None for cam in rtmp_urls}

active_cam = 1
candidate_switch = None
candidate_start_time = None
SWITCH_THRESHOLD = 2.0

def cam_switching(current_cam, ball_coords):
    global active_cam, candidate_switch, candidate_start_time
    if ball_coords is None:
        candidate_switch = None
        candidate_start_time = None
        return active_cam
    x,y = ball_coords
    col = int(x//320)
    row = int(y//360)
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
    if candidate != active_cam:
        now = time.time()
        if candidate == candidate_switch and candidate_start_time and now - candidate_start_time >= SWITCH_THRESHOLD:
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
def frame_reader(cam_id, url):
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"Error: Unable to open stream for cam {cam_id} at {url}")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Cam {cam_id}: frame not received, ending stream.")
            break
        if not frame_queues[cam_id].full():
            frame_queues[cam_id].put(frame)
    cap.release()

def frame_processor(cam_id):
    process_counter = 0
    global active_cam
    while True:
        if not frame_queues[cam_id].empty():
            frame = frame_queues[cam_id].get()
            process_counter += 1

            if cam_id == active_cam:
                proc_frame, ball_coords = run_ball_detector(frame)
                current_active = cam_switching(active_cam, ball_coords)
                stream_label = rtmp_urls[current_active].split("/")[-1]
                final = draw_custom_overlay(proc_frame, action_angle=stream_label)
            else:
                if process_counter % 10 == 0:
                    proc_frame, _ = run_ball_detector(frame)
                    final = proc_frame
                else:
                    final = frame

            latest_frames[cam_id] = final

        time.sleep(0.001)

###############################################################################
# Setup FFmpeg Process for Twitch Streaming
###############################################################################
#if not twitch_stream_key:
#    raise ValueError("TWITCH_STREAM_KEY not found in environment variables.")
twitch_url = f"rtmp://live.twitch.tv/app/{os.getenv('TWITCH_STREAM_KEY')}"

FFMPEG_EXE = r"C:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"

# choose your video encoder flags
if device == "cuda":
    venc = [
        "-c:v", "h264_nvenc",
        "-preset", "p1",         # lowest‐latency NVENC preset
        "-tune",   "ll",         # low‐latency tuning
        "-rc",     "cbr",        # constant‐bitrate mode
    ]
else:
    venc = [
        "-c:v", "libx264",
        "-preset", "ultrafast",  # fastest CPU preset
        "-tune",   "zerolatency",
        "-rc",     "cbr",
    ]

ffmpeg_cmd = [
    FFMPEG_EXE,
    "-y",
    "-loglevel", "info",        # show connection / error messages
    "-re",                       # read at native frame‐rate
    "-f",  "rawvideo",          # video from stdin
    "-pix_fmt", "bgr24",
    "-s",  "1920x1080",
    "-r",  "60",
    "-i",  "-",                 # stdin

    # silent audio (Twitch requires an audio track)
    "-f", "lavfi",
    "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",

    # your video encoder flags
    *venc,

    # keyframe every 1 second
    "-g", "60",
    "-keyint_min", "60",

    # rate control
    "-b:v",     "6000k",
    "-maxrate", "6000k",
    "-bufsize", "6000k",

    # force Twitch-compatible pix_fmt
    "-pix_fmt", "yuv420p",

    # audio encode
    "-c:a", "aac",
    "-b:a", "128k",
    "-ar",  "44100",
    "-ac",  "2",

    # timestamp generation
    "-fflags", "+genpts",

    # output to Twitch
    "-f",  "flv",
    twitch_url
]

ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)


###############################################################################
# Main Function: Spawning Threads, Displaying and Streaming Only the Active Stream
###############################################################################
if __name__ == "__main__":
    reader_threads = []
    processor_threads = []

    # Start reader and processor threads
    for cam_id, url in rtmp_urls.items():
        t_reader = threading.Thread(target=frame_reader, args=(cam_id, url), daemon=True)
        t_processor = threading.Thread(target=frame_processor, args=(cam_id,), daemon=True)
        reader_threads.append(t_reader)
        processor_threads.append(t_processor)
        t_reader.start()
        t_processor.start()

    # Display and stream only the active frame; press 'q' to quit
    while True:
        frame = latest_frames.get(active_cam)
        if frame is not None:
            cv2.imshow("Active Camera Stream", frame)
            try:
                ffmpeg_process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                print("FFmpeg pipe closed. Exiting...")
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()

    # Join threads (they're daemons, so Python will exit anyway)
