import ffmpeg

try:
    input_file = "sample.mp4"
    probe = ffmpeg.probe(input_file)
    print("File information:")
    print(probe)
except ffmpeg.Error as e:
    print("FFmpeg error:")
    print(e.stderr.decode())