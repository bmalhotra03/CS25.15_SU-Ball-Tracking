import numpy as np
import subprocess
from twitchstream.outputvideo import TwitchBufferedOutputStream
from dotenv import load_dotenv
import os

load_dotenv()

# Twitch Streaming Configuration
twitch_stream_key = os.getenv("TWITCH_STREAM_KEY")

ffmpeg_path = os.getenv("FFMPEG_PATH")
if ffmpeg_path:
    os.environ["PATH"] += os.path.pathsep + ffmpeg_path

# Video Configuration
resolution = (1920, 1080)
fps = 30

with TwitchBufferedOutputStream(
    twitch_stream_key=twitch_stream_key,
    width=resolution[0],
    height=resolution[1],
    fps=fps,
    enable_audio=False  # Disable audio
) as stream:
    print("Streaming started. Press Ctrl+C to stop.")
    try:
        while True:
            if stream.get_video_frame_buffer_state() < 30:
                frame = np.random.rand(resolution[1], resolution[0], 3).astype(np.float32)
                stream.send_video_frame(frame)
    except KeyboardInterrupt:
        print("Streaming stopped.")