import sys
import os

# Path to YOLOv7 repository and weights
YOLOV7_PATH = '/Users/hayatesaito/yolov7'  # Path to the YOLOv7 repo
weights = '/Users/hayatesaito/Downloads/yolov7.pt'  # Path to your yolov7.pt file

# Add YOLOV7 path to sys.path dynamically
if YOLOV7_PATH not in sys.path:
    sys.path.append(YOLOV7_PATH)

# Verify the path setup
print(f"YOLOv7 path added to sys.path: {YOLOV7_PATH}")
print(f"Current YOLOv7 Path: {YOLOV7_PATH}")


import cv2
import torch
import subprocess
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device
import torch._dynamo
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


# Device setup (using CPU for now)
device = torch.device('cpu')

# Load YOLOv7 model
model = attempt_load(weights, map_location=device)
model.eval()

# Compiling the model to optimize performance
torch._dynamo.config.suppress_errors = True
model = torch.compile(model, backend="eager")

# Start Webcam (index 1 is used to avoid other device camera)
cap = cv2.VideoCapture(1)


# Define the FFmpeg command to stream to Twitch
stream_key = ''  # Your stream key
twitch_url = f'rtmp://live.twitch.tv/app/{stream_key}'


# FFmpeg command to stream to Twitch
ffmpeg_command = [
    'ffmpeg',  # Command to run FFmpeg
    '-y',  # Overwrite output files without asking for confirmation
    '-f', 'rawvideo',  # Input format for raw video stream (i.e., unprocessed video frames)
    '-vcodec', 'rawvideo',  # Video codec for raw video (no compression)
    '-pix_fmt', 'bgr24',  # Pixel format for the frames. 'bgr24' is standard for OpenCV (3 channels, 8 bits each)
    '-s', '640x360',  # Set video resolution that you like
    '-r', '30',  # Frame rate (30 frames per second)
    '-i', '-',  # Input from stdin (we'll pipe the frames to FFmpeg from the program)
    '-c:v', 'h264_videotoolbox',  # Use the 'libx264' codec for encoding video to H.264 (required for streaming to Twitch) (This is M1 mac unique)
    '-preset', 'veryfast',  # Encoding speed vs. compression trade-off; 'veryfast' means fast encoding with less compression
    '-b:v', '2500k', # Recommentded bitrate for 720p is 2000 - 4000 Kbps
    '-f', 'flv',  # Set the output format to FLV (required for RTMP streaming)
    twitch_url  # The Twitch stream URL where the video will be sent (includes the stream key)
]

# Open a subprocess to run FFmpeg
ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

# Stream Loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from webcam.")
        break

    # Prepare the frame
    img = cv2.resize(frame, (320, 320))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
    img = img.astype('float32') / 255.0
    img = torch.from_numpy(img).unsqueeze(0).to(device)

    # Run YOLOv7 Inference
    with torch.no_grad():
        pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    # Process Results
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in det:
                label = f'{int(cls)} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=(255, 0, 0), line_thickness=2)
    
    
    # Resize frame to match FFmpeg input size (426x240)
    stream_frame = cv2.resize(frame, (640, 360))
    try:
        ffmpeg_process.stdin.write(stream_frame.tobytes())
    except BrokenPipeError:
        print("FFmpeg pipe broken. Stopping stream.")
        break



    # Display the frame locally (optional)
    cv2.imshow("YOLOv7 Webcam", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
ffmpeg_process.stdin.close()
ffmpeg_process.wait()