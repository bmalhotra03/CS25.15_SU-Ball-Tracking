import sys
import os
import dotenv
import numpy as np
from typing import List

dotenv.load_dotenv()

YOLOV7_PATH = os.getenv("YOLOV7_PATH")
if YOLOV7_PATH not in sys.path:
    sys.path.append(YOLOV7_PATH)

import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device
import torch._dynamo
torch._dynamo.config.suppress_errors = True

class BallTracker:
    def __init__(self):
        torch.cuda.empty_cache()
        self.weights = os.getenv("YOLOV7_WEIGHTS")
        self.device = torch.device('cuda')
        self.model = attempt_load(self.weights, map_location=self.device)
        self.model.eval()
        torch._dynamo.config.suppress_errors = True
        self.model = torch.compile(self.model, backend="eager")
        self.length = 320
        self.width = 320
        self.SPORTS_BALL_CLASS_ID = 67

    def locate_ball(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        all_detections = []
        for frame in frames:
            if frame is None:
                print("Frame is None.")
                continue
            # Preprocess the frame
            img = cv2.resize(frame, (self.length, self.width))  # Resize for YOLO input
            img = img[:, :, ::-1].transpose(2, 0, 1)  # Convert BGR to RGB and HWC to CHW
            img = img.astype('float32') / 255.0  # Normalize to [0, 1]
            img = torch.from_numpy(img).unsqueeze(0).to(self.device)  # Convert to tensor

            # Run inference
            with torch.no_grad():
                with torch.autocast("cuda", enabled=torch.cuda.is_available()):  # Enable autocast for faster inference
                    pred = self.model(img, augment=False)[0]
            pred = non_max_suppression(pred, 0.1, 0.45, classes=[self.SPORTS_BALL_CLASS_ID], agnostic=False)

            # Collect results
            frame_detections = []
            for det in pred:
                if det is not None and len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                    
                    for *xyxy, conf, cls in det:
                        detection = {
                            'coordinates': [int(coord) for coord in xyxy],  # Bounding box coordinates
                            'confidence': float(conf),  # Confidence score
                        }
                        frame_detections.append(detection)

            all_detections.append(frame_detections)

        return all_detections