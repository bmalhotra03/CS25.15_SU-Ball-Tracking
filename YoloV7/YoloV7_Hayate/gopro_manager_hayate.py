from open_gopro import WirelessGoPro
import numpy as np
import cv2
import asyncio
import threading
import time
from typing import List
import os
import dotenv
from ball_tracking import BallTracker
from Algorithms.Algorithms import left_or_right, closest_camera


dotenv.load_dotenv()

class GoProManager:
    def __init__(self, ip_addresses):
        # self.ip_addresses = ip_addresses
        self.cameras = [cv2.VideoCapture(0)]
        self.camera_locks = [threading.Lock() for _ in self.cameras]
        self.active_camera_index = 0
        self.last_frame_buffers: List[np.ndarray] = [None] * len(self.cameras)
        self.active_stream_frame_buffer: np.ndarray = None
        self.ball_tracker = BallTracker()
        self.active_stream_lock = threading.Lock()
        self.last_frame_lock = threading.Lock()
        self.frame_stutter = 10
        self.continue_stream = True

        # Twitch stream setup
        # add your twitch stream key in .env file
        self.stream_key = os.getenv("TWITCH_STREAM_KEY")
        self.twitch_url = f'rtmp://live.twitch.tv/app/{self.stream_key}'
        self.ffmpeg_process = None


    def start_processes(self):
        self.stream_webcam_thread = threading.Thread(target=self.stream_webcam)
        self.display_last_frame_buffer_thread = threading.Thread(target=self.display_last_frame_buffer)
        # self.forward_active_stream_frame_buffer_thread = threading.Thread(target=self.forward_active_stream_frame_buffer)
        self.active_camera_controller_thread = threading.Thread(target=self.active_camera_controller)

        self.stream_webcam_thread.start()
        self.display_last_frame_buffer_thread.start()
        # self.forward_active_stream_frame_buffer_thread.start()
        self.active_camera_controller_thread.start()


    def start_ffmpeg_stream(self, width=640, height=360, fps=30, bitrate="2500k"):
        ffmpeg_command = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'h264_videotoolbox',  # use 'libx264' if not on macOS
            '-preset', 'veryfast',
            '-b:v', bitrate,
            '-f', 'flv',
            self.twitch_url
        ]
        self.ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)


    # Thread 1: Accept video feed from webcam and store every x frames of active camera
    # in last_frame_buffers and active_stream_frame_buffer.
    def stream_webcam(self):
        count = 0
        while self.continue_stream:
            ret, frame = self.cameras[self.active_camera_index].read()
            if ret:
                cv2.imshow("Active Stream Buffer", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()
                
                 # Resize for Twitch
                frame_resized = cv2.resize(frame, (640, 360))

                # Send to Twitch
                if self.ffmpeg_process:
                    try:
                        self.ffmpeg_process.stdin.write(frame_resized.tobytes())
                    except (BrokenPipeError, ValueError):
                        print("FFmpeg stream pipe closed unexpectedly.")

                with self.active_stream_lock:
                    self.active_stream_frame_buffer = frame
                if count % self.frame_stutter == 0:
                    with self.last_frame_lock:
                        self.last_frame_buffers[self.active_camera_index] = frame
                count += 1

    # Debugger function
    def display_last_frame_buffer(self):
        while self.continue_stream:
            with self.last_frame_lock:
                last_frame = self.last_frame_buffers[self.active_camera_index]
            if last_frame is not None:
                # Display the last frame buffer
                cv2.imshow("Last Frame Buffer", last_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
            time.sleep(0.1)

    # Thread 2: Forward active_stream_frame_buffer to ball_tracker.
    def forward_active_stream_frame_buffer(self):
        while self.continue_stream:
            pass

    # Thread 3: Continously loops to locate phone in each (and only one) frame in last_frame_buffers from ball_tracker and then
    # calls left_or_right to determine if the phone is on the left or right side of the frame and prints the result.
    def active_camera_controller(self):
        while self.continue_stream:
            last_frames: List[np.ndarray] = []
            for i in range(len(self.last_frame_buffers)):
                with self.camera_locks[i]:
                    last_frames.append(self.last_frame_buffers[i])
            
            if len(last_frames) > 0:
                detections = self.ball_tracker.locate_ball(last_frames)
                if len(detections) > 0:
                    coordinates = [
                        [detection['coordinates'] for detection in detection_list if detection['coordinates'] is not None]
                        for detection_list in detections
                    ]
                    if len(coordinates) > 0:
                        asyncio.run(self.print_closest_camera(last_frames[0], coordinates))
            else:
                print("Last frame is None.")

    async def print_closest_camera(self, frame, ball_coordinates):
        position = await closest_camera(frame, ball_coordinates)
        print(position)

    def kill_stream_controller(self):
        print("Stopping all streams...")
        self.continue_stream = False
        for camera in self.cameras:
            camera.release()
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.wait()
            except Exception as e:
                print(f"Error closing FFmpeg: {e}")
            print("FFmpeg stream ended.")
        self.stream_webcam_thread.join()
        cv2.destroyAllWindows()
        print("All resources released.")

    
        