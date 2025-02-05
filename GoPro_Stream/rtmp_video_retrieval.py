import cv2
import threading

# Define RTMP URLs for the four GoPro streams
rtmp_urls = [
    "rtmp://192.168.1.100/live/GoPro_SU1",
    "rtmp://192.168.1.100/live/GoPro_SU2",
    "rtmp://192.168.1.100/live/GoPro_SU3",
    "rtmp://192.168.1.100/live/GoPro_SU4"
]

# Function to handle the video stream from one source
def stream_video(rtmp_url, window_name):
    cap = cv2.VideoCapture(rtmp_url)

    if not cap.isOpened():
        print(f"Error: Unable to open the RTMP stream for {window_name}")
        return

    print(f"Streaming from {window_name}... Press 'q' in any window to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Unable to grab frame from {window_name}")
            break

        # Display the frame in an OpenCV window
        cv2.imshow(window_name, frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(window_name)

# Start threads for each stream
threads = []
for i, url in enumerate(rtmp_urls):
    thread = threading.Thread(target=stream_video, args=(url, f"GoPro {i+1}"))
    thread.start()
    threads.append(thread)

# Wait for all threads to finish
for thread in threads:
    thread.join()

cv2.destroyAllWindows()
