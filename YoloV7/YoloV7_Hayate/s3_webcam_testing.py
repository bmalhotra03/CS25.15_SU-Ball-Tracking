import sys
import os
import io
from datetime import datetime


# Path to YOLOv7 repository and weights
YOLOV7_PATH = '/Users/hayatesaito/yolov7'  # Path to the YOLOv7 repo
weights = '/Users/hayatesaito/Downloads/yolov7.pt'  # Path to your yolov7.pt file

# Add YOLOV7 path to sys.path dynamically
if YOLOV7_PATH not in sys.path:
    sys.path.append(YOLOV7_PATH)

# Verify the path setup
print(f"YOLOv7 path added to sys.path: {YOLOV7_PATH}")
print(f"Current YOLOv7 Path: {YOLOV7_PATH}")


# YOLOv7 Imports
import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device
import boto3
import torch._dynamo

# It suppresses unnecessary UserWarning messages from PyTorch and OpenCV to keep the console output clean.
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


device = torch.device('cpu') # my computer (M1 Mac) can't use CUDA for some reason

# Load Model
model = attempt_load(weights, map_location=device)
model.eval()

# We need to put crednetials to use boto3 with our AWS account
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    region_name='us-west-2'
)

# I created the bucket in Oregeon region (us-west-2)
bucket_name = 'cs-25-15-frames'


# Suppresses errors related to model compilation.
# Compiles the model using the eager backend, optimizing for performance.
torch._dynamo.config.suppress_errors = True
model = torch.compile(model, backend="eager")


# Start Webcam (For some reason, index 0 is using my other devices's camera)
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("❌ Could not open webcam. Try changing the index (0, 1, 2...)")
    exit()

# Get frame properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS) or 24)

# Create local video file name
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_filename = f"yolov7_session_{timestamp}.mp4"

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

print("🎥 Recording started — press 'q' to stop.")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the frame
    img = cv2.resize(frame, (320, 320))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = img.astype('float32') / 255.0
    img = torch.from_numpy(img).unsqueeze(0).to(device)

    # Run Inference
    with torch.no_grad():
        with torch.autocast("cuda"):
            pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    # Process Results
    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in det:
                label = f'{int(cls)} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=(255, 0, 0), line_thickness=2)

    # Save frame to video file
    out.write(frame)


    # Display
    cv2.imshow("YOLOv7 Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Video saved locally: {video_filename}")

# Upload to S3
s3_key = f"videos/{video_filename}"
try:
    s3.upload_file(video_filename, bucket_name, s3_key)
    print(f"✅ Uploaded to S3: {s3_key}")
    os.remove(video_filename)
    print(f"🗑️ Deleted local file: {video_filename}")
except Exception as e:
    print(f"❌ Upload failed: {e}")