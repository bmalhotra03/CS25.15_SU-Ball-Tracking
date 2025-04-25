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
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device

###############################################################################
# Environment Setup & YOLOv7 Model Loading
###############################################################################
dotenv.load_dotenv()

YOLOV7_PATH    = os.getenv("YOLOV7_PATH")
YOLOV7_WEIGHTS = os.getenv("YOLOV7_WEIGHTS", "yolov7.pt")
if YOLOV7_PATH and YOLOV7_PATH not in sys.path:
    sys.path.append(YOLOV7_PATH)

from models.experimental import attempt_load

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
    except Exception:
        pass
    model.half()

SPORTS_BALL_CLASS_ID = 32
CONF_THRESHOLD       = 0.5
NMS_THRESHOLD        = 0.3

###############################################################################
# Overlay Setup: Assets and Helper Functions
###############################################################################
tmobile_logo  = cv2.imread("assets/Tmobile_Logo.png")
seattle_logo  = cv2.imread("assets/SeattleU_SponsorLogo.png")
home_logo     = cv2.imread("assets/SeattleU_Logo.png")
away_logo     = cv2.imread("assets/UW_Logo.png")

if tmobile_logo is not None: tmobile_logo = cv2.resize(tmobile_logo, (80, 80))
if seattle_logo is not None: seattle_logo = cv2.resize(seattle_logo, (80, 80))
if home_logo    is not None: home_logo    = cv2.resize(home_logo,    (40, 40))
if away_logo    is not None: away_logo    = cv2.resize(away_logo,    (40, 40))

# Preload fonts just once (instead of per-frame)
FONT_PATH        = "assets/FuturaMaxi.otf"
ANGLE_FONT_PATH  = "assets/FuturaMaxi.otf"
FONT_SIZE        = 16
ANGLE_FONT_SIZE  = 12
TITLE_FONT       = ImageFont.truetype(FONT_PATH,        FONT_SIZE)
INFO_FONT        = ImageFont.truetype(ANGLE_FONT_PATH, ANGLE_FONT_SIZE)

def draw_pil_text(img, text, x, y, w, h, font, text_color=(0,0,0)):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    bbox = draw.textbbox((0,0), text, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    tx = x + (w-tw)//2
    ty = y + (h-th)//2 + 2
    draw.text((tx, ty), text, font=font, fill=text_color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def draw_image(frame, image, x, y, w, h):
    if image is None: return
    frame[y:y+h, x:x+w] = image

def draw_acronym_score_box(frame, acronym, score, x, y, w, h, font):
    cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255), thickness=-1)
    mid = x + w//2
    cv2.line(frame, (mid,y), (mid,y+h), (0,0,0), 2)
    frame = draw_pil_text(frame, acronym, x, y, w//2, h, font)
    return draw_pil_text(frame, score, mid, y, w//2, h, font)

def draw_custom_overlay(frame, home_acronym="HOM", away_acronym="AWY",
                        home_score="0", away_score="0", action_angle=""):
    font       = TITLE_FONT
    angle_font = INFO_FONT

    # logos
    draw_image(frame, tmobile_logo, 10, 10, 80, 80)
    draw_image(frame, seattle_logo, 90, 10, 80, 80)

    # home row
    draw_image(frame, home_logo, 170, 10, 40, 40)
    frame = draw_acronym_score_box(frame, home_acronym, home_score, 210, 10, 80, 40, font)

    # away row
    draw_image(frame, away_logo, 170, 50, 40, 40)
    frame = draw_acronym_score_box(frame, away_acronym, away_score, 210, 50, 80, 40, font)

    # info row
    info_x, info_y, info_w, info_h = 10, 90, 280, 30
    cv2.rectangle(frame, (info_x, info_y), (info_x+info_w, info_y+info_h), (255,255,255), thickness=-1)
    return draw_pil_text(frame,
                         f"CS 25.15  |  ACTION ANGLE: {action_angle}",
                         info_x, info_y, info_w, info_h, angle_font)

###############################################################################
# Ball Detection Helper Functions (modified to also return ball coordinates)
###############################################################################
def letterbox(img, new_shape=(640,640), color=(114,114,114)):
    h0, w0 = img.shape[:2]
    r = min(new_shape[0]/h0, new_shape[1]/w0)
    new_unpad = (int(w0*r), int(h0*r))
    dw, dh = (new_shape[1]-new_unpad[0])//2, (new_shape[0]-new_unpad[1])//2
    resized = cv2.resize(img, new_unpad)
    return cv2.copyMakeBorder(resized, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)

def run_ball_detector(frame):
    orig = frame.copy()
    img = letterbox(frame)
    img = img[:, :, ::-1].transpose(2,0,1)
    img = np.ascontiguousarray(img, dtype=np.float32)/255.0
    timg = torch.from_numpy(img).unsqueeze(0).to(device)
    if device=="cuda": timg = timg.half()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device=="cuda")):
        pred = model(timg, augment=False)[0]
    det = non_max_suppression(pred, CONF_THRESHOLD, NMS_THRESHOLD,
                              classes=[SPORTS_BALL_CLASS_ID])[0]
    ball_coords = None
    if det is not None and len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orig.shape).round()
        x1,y1,x2,y2,conf,_ = det[0]
        plot_one_box([x1,y1,x2,y2], orig, label=f"{conf:.2f}", color=(0,255,0), line_thickness=2)
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
frame_queues   = {cam: queue.Queue(maxsize=5) for cam in rtmp_urls}
latest_frames  = {cam: None for cam in rtmp_urls}
active_cam     = 1
candidate_switch    = None
candidate_start_time = None
SWITCH_THRESHOLD     = 2.0  # seconds

def cam_switching(current_cam, ball_coords):
    global active_cam, candidate_switch, candidate_start_time
    if ball_coords is None:
        candidate_switch    = None
        candidate_start_time = None
        return active_cam
    x,y = ball_coords
    col, row = int(x//320), int(y//360)
    candidate = current_cam
    if row == 0:
        if current_cam == 1: candidate = 4 if col==5 else 2
        elif current_cam == 2: candidate = 3 if col==0 else 1
        elif current_cam == 3: candidate = 2 if col==0 else 4
        elif current_cam == 4: candidate = 1 if col==5 else 3
    if candidate != active_cam:
        now = time.time()
        if candidate_switch == candidate and candidate_start_time and now - candidate_start_time >= SWITCH_THRESHOLD:
            print(f"Switching active stream from cam {active_cam} to cam {candidate}")
            active_cam = candidate
            candidate_switch    = None
            candidate_start_time = None
        else:
            candidate_switch    = candidate
            candidate_start_time = now
    else:
        candidate_switch    = None
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
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Cam {cam_id}: frame not received, ending stream.")
            break
        if not frame_queues[cam_id].full():
            frame_queues[cam_id].put(frame)
    cap.release()

def frame_processor(cam_id, out_queue):
    global active_cam
    counter = 0
    while True:
        if not frame_queues[cam_id].empty():
            frame = frame_queues[cam_id].get()
            counter += 1
            if cam_id == active_cam:
                proc, coords = run_ball_detector(frame)
                active_cam = cam_switching(active_cam, coords)
                label = rtmp_urls[active_cam].split("/")[-1]
                final = draw_custom_overlay(proc, action_angle=label)
            else:
                if counter % 10 == 0:
                    proc, _ = run_ball_detector(frame)
                    final = proc
                else:
                    final = frame
            latest_frames[cam_id] = final
            try:
                out_queue.put_nowait(final.tobytes())
            except queue.Full:
                pass
        time.sleep(0.001)

###############################################################################
# Setup FFmpeg Process for Twitch Streaming
###############################################################################
twitch_stream_key = os.getenv("TWITCH_STREAM_KEY")
if not twitch_stream_key:
    raise ValueError("TWITCH_STREAM_KEY not found in environment variables.")
twitch_url = f"rtmps://live.twitch.tv:443/app/{twitch_stream_key}"
FFMPEG_EXE = r"C:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"

# ffmpeg_cmd = [
#     FFMPEG_EXE,
#     "-y",                       # overwrite output
#     # → VIDEO input from raw frames
#     "-f",   "rawvideo",
#     "-pix_fmt", "bgr24",
#     "-s",   "1920x1080",
#     "-r",   "60",
#     "-i",   "-",               # read video from stdin
#
#     # → AUDIO input: silent stereo track (Twitch requires audio)
#     "-f",  "lavfi",
#     "-i",  "anullsrc=channel_layout=stereo:sample_rate=44100",
#
#     # → VIDEO encoding (NVIDIA NVENC, low-latency preset)
#     "-c:v",     "h264_nvenc",
#     "-preset",  "llhp",
#     "-tune",    "ll",
#
#     # → Keyframe interval: 1 s GOP
#     "-g",         "60",
#     "-keyint_min", "60",
#
#     # → Bitrate control
#     "-b:v",     "6000k",
#     "-maxrate", "6000k",
#     "-bufsize", "6000k",
#
#     # → (Optional) reduce internal buffering
#     "-fflags",        "nobuffer",
#     "-flags",         "low_delay",
#     "-flush_packets", "0",
#     "-fflags",        "+genpts",
#
#     # → AUDIO encoding
#     "-c:a",   "aac",
#     "-b:a",   "128k",
#
#     # → RTMP output
#     "-f",   "flv",
#     twitch_url
# ]

# Pick GPU or CPU encoder
if device == "cuda":
    venc = ["-c:v", "h264_nvenc", "-preset", "llhp", "-tune", "ll"]
else:
    venc = ["-c:v", "libx264",  "-preset", "ultrafast", "-tune", "zerolatency"]

ffmpeg_cmd = [
    FFMPEG_EXE, "-y",
    "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1920x1080", "-r", "60", "-i", "-",
    "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
    *venc,
    "-g", "60", "-keyint_min", "60",
    "-b:v", "6000k", "-maxrate", "6000k", "-bufsize", "6000k",
    "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
    "-fflags", "+genpts",
    "-f", "flv", twitch_url
]

def ffmpeg_writer(in_queue):
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    while True:
        data = in_queue.get()
        try:
            proc.stdin.write(data)
        except BrokenPipeError:
            break

###############################################################################
# Main Function: Spawning Threads, Displaying and Streaming Only the Active Stream
###############################################################################
if __name__ == "__main__":
    out_queue = queue.Queue(maxsize=10)

    # start FFmpeg writer thread
    threading.Thread(target=ffmpeg_writer, args=(out_queue,), daemon=True).start()

    # start reader & processor threads
    for cam_id, url in rtmp_urls.items():
        threading.Thread(target=frame_reader,   args=(cam_id, url),        daemon=True).start()
        threading.Thread(target=frame_processor, args=(cam_id, out_queue), daemon=True).start()

    # display loop
    while True:
        frame = latest_frames.get(active_cam)
        if frame is not None:
            cv2.imshow("Active Camera Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()