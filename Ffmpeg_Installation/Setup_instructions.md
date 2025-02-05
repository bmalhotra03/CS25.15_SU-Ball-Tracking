## Yolo Setup Instructions

1. Install ffmpeg: https://ffmpeg.org/download.html
2. Ensure you download the prebuilt binaries for your system, not the source code. If there's no bin folder, you've downloaded the source code.
3. Extract the file.
4. Add the ffmpeg bin folder path to the system path. On Windows, do this through environment variables. On Linux, add it to the .bashrc file.
5. Verify the installation by running `ffmpeg -version` in the command line.
6. If you get an error, you probably didn't add the path to the system path.
7. Try running the ffmpeg_test.py file to see if it works.
8. If ffmpeg_test.py doesn't work, but 'ffmpeg -version' does, close your IDE and reopen it.