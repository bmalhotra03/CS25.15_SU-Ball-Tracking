import subprocess
import os
import dotenv

dotenv.load_dotenv()


def start_ffmpeg_process(width=1280, height=720, fps=30, pix_fmt="bgr24", twitch_url=os.getenv("TWITCH_URL")):
    """
    Launches FFmpeg in such a way that it reads raw frames from stdin (pipe:0)
    and pushes them to the Twitch RTMP URL.
    """
    cmd = [
        "ffmpeg",
        "-f", "rawvideo",
        "-pix_fmt", pix_fmt,
        "-s", f"{width}x{height}",   # image size
        "-r", str(fps),              # frame rate
        "-i", "pipe:0",             # read from stdin
        # video encoding:
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-b:v", "2500k",
        "-maxrate", "2500k",
        "-bufsize", "5000k",
        "-g", "50",
        # output format and URL:
        "-f", "flv",
        twitch_url
    ]

    # Popen with stdin as a PIPE—meaning we can write frames into ffmpeg_proc.stdin
    ffmpeg_proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return ffmpeg_proc

