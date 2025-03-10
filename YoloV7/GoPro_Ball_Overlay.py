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
# Input size for ball detection
INPUT_SIZE = 320

# (Optional) Create a lock if you experience threading issues with the model
model_lock = threading.Lock()

def run_ball_detector(frame):
    """
    Process a frame using YOLOv7 to detect ONLY a soccer ball.
    Preprocess the frame to 320x320, run inference with class filter (67),
    and draw bounding boxes on the original frame.
    """
    orig_frame = frame.copy()
    # Resize frame to 320x320
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    # Convert BGR to RGB and change data layout from HWC to CHW
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = img.astype('float32') / 255.0  # Normalize
    img = torch.from_numpy(img).unsqueeze(0).to(device)

    # Run inference with torch.autocast if available for speed
    with torch.no_grad():
        with torch.autocast("cuda", enabled=(device == "cuda")):
            with model_lock:
                pred = model(img, augment=False)[0]

    # Apply non-max suppression limiting detections to the sports ball class
    pred = non_max_suppression(
        pred,
        conf_thres=0.1,
        iou_thres=0.45,
        classes=[SPORTS_BALL_CLASS_ID],
        agnostic=False
    )

    # Process detections and draw boxes
    for det in pred:
        if det is not None and len(det):
            # Rescale coordinates to the original frame size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orig_frame.shape).round()
            for *xyxy, conf, cls in det:
                label = f'soccer ball {conf:.2f}'
                # Draw bounding box in green
                plot_one_box(xyxy, orig_frame, label=label, color=(0, 255, 0), line_thickness=2)

    return orig_frame

def draw_scoreboard(
    frame,
    tmobile_logo=None,
    seattle_logo=None,
    home_logo=None,
    away_logo=None,
    home_acronym="HOM",
    away_acronym="AWY",
    home_score=0,
    away_score=0,
    action_angle="GoPro_SU1"
):
    """
    Draws a 2-row scoreboard on a red background in the top-left corner:
      Row 1: T-Mobile logo, Home logo, Home acronym, Home score
      Row 2: Seattle U logo, Away logo, Away acronym, Away score
    Then, in the top-right corner, draws "Action Angle: [action_angle]".
    """

    # Basic scoreboard sizing/position
    sb_x, sb_y = 10, 10
    sb_width, sb_height = 450, 100  # Adjust as needed

    # Draw red background for scoreboard
    red_color = (0, 0, 255)  # BGR: red
    cv2.rectangle(
        frame,
        (sb_x, sb_y),
        (sb_x + sb_width, sb_y + sb_height),
        red_color,
        thickness=-1
    )

    # Each row's vertical start
    row1_y = sb_y + 10
    row2_y = sb_y + 55  # second row ~45-50 px below row1

    # Common spacing values
    logo_size = 35  # logos will be 35×35
    spacing = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    text_color = (0, 0, 0)  # black text
    text_thickness = 2      # thicker text for a “bold” look

    # -------------------------
    # Row 1: T-Mobile | Home Logo | Home Acronym | Home Score
    # -------------------------
    # 1) T-Mobile logo
    if tmobile_logo is not None:
        tmobile_logo_resized = cv2.resize(tmobile_logo, (logo_size, logo_size))
        frame[row1_y:row1_y+logo_size, sb_x+spacing:sb_x+spacing+logo_size] = tmobile_logo_resized

    # 2) Home logo
    home_logo_x = sb_x + spacing + logo_size + spacing
    if home_logo is not None:
        home_logo_resized = cv2.resize(home_logo, (logo_size, logo_size))
        frame[row1_y:row1_y+logo_size, home_logo_x:home_logo_x+logo_size] = home_logo_resized

    # 3) Home acronym
    home_acronym_x = home_logo_x + logo_size + spacing
    cv2.putText(
        frame,
        home_acronym,
        (home_acronym_x, row1_y + logo_size - 10),
        font,
        font_scale,
        text_color,
        text_thickness,
        cv2.LINE_AA
    )

    # 4) Home score
    home_score_x = home_acronym_x + 70
    cv2.putText(
        frame,
        str(home_score),
        (home_score_x, row1_y + logo_size - 10),
        font,
        font_scale,
        text_color,
        text_thickness,
        cv2.LINE_AA
    )

    # -------------------------
    # Row 2: Seattle U | Away Logo | Away Acronym | Away Score
    # -------------------------
    # 1) Seattle U logo
    if seattle_logo is not None:
        seattle_logo_resized = cv2.resize(seattle_logo, (logo_size, logo_size))
        frame[row2_y:row2_y+logo_size, sb_x+spacing:sb_x+spacing+logo_size] = seattle_logo_resized

    # 2) Away logo
    away_logo_x = sb_x + spacing + logo_size + spacing
    if away_logo is not None:
        away_logo_resized = cv2.resize(away_logo, (logo_size, logo_size))
        frame[row2_y:row2_y+logo_size, away_logo_x:away_logo_x+logo_size] = away_logo_resized

    # 3) Away acronym
    away_acronym_x = away_logo_x + logo_size + spacing
    cv2.putText(
        frame,
        away_acronym,
        (away_acronym_x, row2_y + logo_size - 10),
        font,
        font_scale,
        text_color,
        text_thickness,
        cv2.LINE_AA
    )

    # 4) Away score
    away_score_x = away_acronym_x + 70
    cv2.putText(
        frame,
        str(away_score),
        (away_score_x, row2_y + logo_size - 10),
        font,
        font_scale,
        text_color,
        text_thickness,
        cv2.LINE_AA
    )

    # --------------------------------
    # Top-right "Action Angle" box
    # --------------------------------
    frame_h, frame_w = frame.shape[:2]
    aa_width, aa_height = 250, 40
    aa_x = frame_w - aa_width - 10
    aa_y = 10

    # White rectangle behind text (optional)
    cv2.rectangle(
        frame,
        (aa_x, aa_y),
        (aa_x + aa_width, aa_y + aa_height),
        (255, 255, 255),
        thickness=-1
    )

    # Action Angle text
    action_text = f"Action Angle: {action_angle}"
    cv2.putText(
        frame,
        action_text,
        (aa_x + 10, aa_y + 25),
        font,
        0.7,
        (0, 0, 0),  # black
        2,
        cv2.LINE_AA
    )

    return frame

def stream_video(rtmp_url, window_name, tmobile_logo, seattle_logo, home_logo, away_logo,
                 home_acronym="SEA", away_acronym="POR"):
    """
    Opens an RTMP stream, runs YOLO-based soccer ball detection on each frame,
    and overlays a scoreboard in the top-left and "Action Angle" in the top-right.
    """
    cap = cv2.VideoCapture(rtmp_url)
    if not cap.isOpened():
        print(f"Error: Unable to open the RTMP stream for {window_name}")
        return

    # Set resolution to Full HD (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Extract something like "GoPro_SU1" from the RTMP URL for Action Angle
    action_label = rtmp_url.split('/')[-1]  # e.g., "GoPro_SU1"

    print(f"Streaming from {window_name} with soccer ball detection... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Unable to grab frame from {window_name}")
            break

        # 1) Detect soccer ball
        processed_frame = run_ball_detector(frame)

        # 2) Draw scoreboard
        scoreboard_frame = draw_scoreboard(
            processed_frame,
            tmobile_logo=tmobile_logo,
            seattle_logo=seattle_logo,
            home_logo=home_logo,
            away_logo=away_logo,
            home_acronym=home_acronym,
            away_acronym=away_acronym,
            home_score=0,  # Example placeholder
            away_score=0,  # Example placeholder
            action_angle=action_label
        )

        # Show final frame
        cv2.imshow(window_name, scoreboard_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(window_name)

if __name__ == "__main__":
    # List of RTMP stream URLs
    rtmp_urls = [
        "rtmp://192.168.1.100/live/GoPro_SU1",
        "rtmp://192.168.1.100/live/GoPro_SU2",
        "rtmp://192.168.1.100/live/GoPro_SU3",
        "rtmp://192.168.1.100/live/GoPro_SU4"
    ]

    # Load logos (replace these paths with your actual image files)
    # If you don't have a logo, set it to None
    tmobile_logo = cv2.imread("assets/tmobile.png")
    seattle_logo = cv2.imread("assets/seattleu.png")
    home_logo = cv2.imread("assets/home_team.png")
    away_logo = cv2.imread("assets/away_team.png")

    # Start a thread for each RTMP stream
    threads = []
    for i, url in enumerate(rtmp_urls):
        # You can customize acronyms or scores here, or pass them in dynamically
        thread = threading.Thread(
            target=stream_video,
            args=(
                url,
                f"GoPro {i+1}",
                tmobile_logo,
                seattle_logo,
                home_logo,
                away_logo,
                "SEA",  # Home acronym
                "POR"   # Away acronym
            )
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    cv2.destroyAllWindows()