import sys
import os
import dotenv

dotenv.load_dotenv()

# Add yolov7 path to sys.path dynamically
YOLOV7_PATH = os.getenv("YOLOV7_PATH")
if YOLOV7_PATH not in sys.path:
    sys.path.append(YOLOV7_PATH)

# Verify the path setup
print(f"YOLOv7 path added to sys.path: {YOLOV7_PATH}")

# YOLOv7 Imports
import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device
import torch._dynamo


torch.cuda.empty_cache()

# Initialize
weights = os.getenv("YOLOV7_WEIGHTS")
device = torch.device('cuda')

# Load Model
model = attempt_load(weights, map_location=device)
model.eval()

torch._dynamo.config.suppress_errors = True
model = torch.compile(model, backend="eager")

# Start Webcam
cap = cv2.VideoCapture(0)

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

    # Display
    cv2.imshow("YOLOv7 Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()