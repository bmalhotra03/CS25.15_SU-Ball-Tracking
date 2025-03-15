import cv2
import multiprocessing
import os
import dotenv
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device

###############################################################################
# Environment Setup & YOLOv7 Model Loading
###############################################################################
dotenv.load_dotenv()

YOLOV7_PATH = os.getenv("YOLOV7_PATH")
YOLOV7_WEIGHTS = os.getenv("YOLOV7_WEIGHTS", "yolov7.pt")

if YOLOV7_PATH and YOLOV7_PATH not in sys.path:
    sys.path.append(YOLOV7_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
if not os.path.exists(YOLOV7_WEIGHTS):
    raise FileNotFoundError(f"Model weights not found: {YOLOV7_WEIGHTS}")

print("Loading YOLOv7 model...")
model = attempt_load(YOLOV7_WEIGHTS, map_location=device)
model.eval()

if device == "cuda":
    model.half()

SPORTS_BALL_CLASS_ID = 32
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3

###############################################################################
# Overlay Setup: Assets and Helper Functions
###############################################################################
assets_path = "assets/"
tmobile_logo = cv2.imread(os.path.join(assets_path, "Tmobile_Logo.png"))
seattle_logo = cv2.imread(os.path.join(assets_path, "SeattleU_SponsorLogo.png"))
home_logo = cv2.imread(os.path.join(assets_path, "SeattleU_Logo.png"))
away_logo = cv2.imread(os.path.join(assets_path, "UW_Logo.png"))

if tmobile_logo is not None:
    tmobile_logo = cv2.resize(tmobile_logo, (80, 80))
if seattle_logo is not None:
    seattle_logo = cv2.resize(seattle_logo, (80, 80))
if home_logo is not None:
    home_logo = cv2.resize(home_logo, (40, 40))
if away_logo is not None:
    away_logo = cv2.resize(away_logo, (40, 40))

FONT_PATH = os.path.join(assets_path, "FuturaMaxi.otf")

def load_font(size):
    return ImageFont.truetype(FONT_PATH, size)

def draw_pil_text(img, text, x, y, w, h, font, text_color=(0, 0, 0)):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    centered_x = x + (w - text_w) // 2
    centered_y = y + (h - text_h) // 2 + 2
    draw.text((centered_x, centered_y), text, font=font, fill=text_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def draw_overlay(frame, home_score="0", away_score="0", action_angle=""):
    font = load_font(16)
    angle_font = load_font(12)

    scoreboard_x, scoreboard_y = 10, 10
    frame[scoreboard_y:scoreboard_y + 80, scoreboard_x:scoreboard_x + 80] = tmobile_logo
    frame[scoreboard_y:scoreboard_y + 80, scoreboard_x + 80:scoreboard_x + 160] = seattle_logo

    home_row_x, home_row_y = scoreboard_x + 160, scoreboard_y
    frame[home_row_y:home_row_y + 40, home_row_x:home_row_x + 40] = home_logo
    frame = draw_pil_text(frame, f"SU: {home_score}", home_row_x + 50, home_row_y, 80, 40, font)

    away_row_x, away_row_y = scoreboard_x + 160, scoreboard_y + 40
    frame[away_row_y:away_row_y + 40, away_row_x:away_row_x + 40] = away_logo
    frame = draw_pil_text(frame, f"UW: {away_score}", away_row_x + 50, away_row_y, 80, 40, font)

    info_text = f"CS 25.15  |  ACTION ANGLE: {action_angle}"
    frame = draw_pil_text(frame, info_text, scoreboard_x, away_row_y + 50, 280, 30, angle_font)

    return frame

###############################################################################
# Ball Detection with YOLOv7
###############################################################################
def letterbox(img, new_shape=(640, 640)):
    shape = img.shape[:2]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw // 2, dh // 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img

def run_ball_detector(frame):
    img = letterbox(frame, new_shape=(640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img = torch.from_numpy(img).unsqueeze(0).to(device)

    if device == "cuda":
        img = img.half()

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device=="cuda")):
            pred = model(img, augment=False)[0]

    pred = non_max_suppression(pred, conf_thres=CONF_THRESHOLD, iou_thres=NMS_THRESHOLD,
                               classes=[SPORTS_BALL_CLASS_ID], agnostic=False)

    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                label = f'Soccer Ball {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=(0, 255, 0), line_thickness=2)
    return frame

###############################################################################
# Multiprocessing Setup
###############################################################################
def frame_reader(rtmp_url, frame_queue):
    cap = cv2.VideoCapture(rtmp_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_queue.full():
            frame_queue.get()
        frame_queue.put(frame)

    cap.release()

def frame_processor(frame_queue, window_name):
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            processed_frame = run_ball_detector(frame)
            final_frame = draw_overlay(processed_frame, action_angle=window_name)
            cv2.imshow(window_name, final_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    streams = ["rtmp://192.168.1.100/live/GoPro1"]
    processes = []

    for stream in streams:
        queue = multiprocessing.Queue(maxsize=5)
        processes.append(multiprocessing.Process(target=frame_reader, args=(stream, queue)))
        processes.append(multiprocessing.Process(target=frame_processor, args=(queue, stream)))

    for p in processes:
        p.start()
    for p in processes:
        p.join()