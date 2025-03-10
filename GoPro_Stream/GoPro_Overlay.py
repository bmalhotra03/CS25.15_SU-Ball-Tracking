import cv2
import threading
import os

# Define RTMP URLs for the four GoPro streams
rtmp_urls = [
    "rtmp://192.168.1.100/live/GoPro_SU1",
    "rtmp://192.168.1.100/live/GoPro_SU2",
    "rtmp://192.168.1.100/live/GoPro_SU3",
    "rtmp://192.168.1.100/live/GoPro_SU4"
]

# Load logos (replace these paths with your actual image files)
# If a logo is missing, set it to None
tmobile_logo = cv2.imread("assets/Tmobile_Logo.png")
seattle_logo = cv2.imread("assets/SeattleU_Logo.png")
home_logo = cv2.imread("assets/SeattleU_Logo.png")
away_logo = cv2.imread("assets/SeattleU_Logo.png")

def draw_scoreboard(
    frame,
    tmobile_logo=None,
    seattle_logo=None,
    home_logo=None,
    away_logo=None,
    home_acronym="HOM",
    away_acronym="AWY",
    home_score=0,
    away_score=0,
    action_angle="GoPro_SU1"
):
    """
    Draws a scoreboard overlay in the top-left corner:
      Row 1: T-Mobile logo, Home logo, Home acronym, Home score
      Row 2: Seattle U logo, Away logo, Away acronym, Away score
    Also draws "Action Angle: [GoPro Name]" in the top-right.
    """

    # Scoreboard positioning
    sb_x, sb_y = 10, 10
    sb_width, sb_height = 450, 100  # Adjust as needed

    # Draw red background for scoreboard
    red_color = (0, 0, 255)  # Red (BGR)
    cv2.rectangle(frame, (sb_x, sb_y), (sb_x + sb_width, sb_y + sb_height), red_color, thickness=-1)

    # Row spacing
    row1_y = sb_y + 10
    row2_y = sb_y + 55
    logo_size = 35  # Standard logo size
    spacing = 10  # Space between elements
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    text_color = (0, 0, 0)  # Black text
    text_thickness = 2  # Bold

    # -------------------------
    # Row 1: T-Mobile | Home Logo | Home Acronym | Home Score
    # -------------------------
    # 1) T-Mobile logo
    if tmobile_logo is not None:
        tmobile_logo_resized = cv2.resize(tmobile_logo, (logo_size, logo_size))
        frame[row1_y:row1_y+logo_size, sb_x+spacing:sb_x+spacing+logo_size] = tmobile_logo_resized

    # 2) Home logo
    home_logo_x = sb_x + spacing + logo_size + spacing
    if home_logo is not None:
        home_logo_resized = cv2.resize(home_logo, (logo_size, logo_size))
        frame[row1_y:row1_y+logo_size, home_logo_x:home_logo_x+logo_size] = home_logo_resized

    # 3) Home acronym
    home_acronym_x = home_logo_x + logo_size + spacing
    cv2.putText(frame, home_acronym, (home_acronym_x, row1_y + logo_size - 10), font, font_scale, text_color, text_thickness, cv2.LINE_AA)

    # 4) Home score
    home_score_x = home_acronym_x + 70
    cv2.putText(frame, str(home_score), (home_score_x, row1_y + logo_size - 10), font, font_scale, text_color, text_thickness, cv2.LINE_AA)

    # -------------------------
    # Row 2: Seattle U | Away Logo | Away Acronym | Away Score
    # -------------------------
    # 1) Seattle U logo
    if seattle_logo is not None:
        seattle_logo_resized = cv2.resize(seattle_logo, (logo_size, logo_size))
        frame[row2_y:row2_y+logo_size, sb_x+spacing:sb_x+spacing+logo_size] = seattle_logo_resized

    # 2) Away logo
    away_logo_x = sb_x + spacing + logo_size + spacing
    if away_logo is not None:
        away_logo_resized = cv2.resize(away_logo, (logo_size, logo_size))
        frame[row2_y:row2_y+logo_size, away_logo_x:away_logo_x+logo_size] = away_logo_resized

    # 3) Away acronym
    away_acronym_x = away_logo_x + logo_size + spacing
    cv2.putText(frame, away_acronym, (away_acronym_x, row2_y + logo_size - 10), font, font_scale, text_color, text_thickness, cv2.LINE_AA)

    # 4) Away score
    away_score_x = away_acronym_x + 70
    cv2.putText(frame, str(away_score), (away_score_x, row2_y + logo_size - 10), font, font_scale, text_color, text_thickness, cv2.LINE_AA)

    # --------------------------------
    # Top-right "Action Angle" box
    # --------------------------------
    frame_h, frame_w = frame.shape[:2]
    aa_width, aa_height = 250, 40
    aa_x = frame_w - aa_width - 10
    aa_y = 10

    # White rectangle for "Action Angle"
    cv2.rectangle(frame, (aa_x, aa_y), (aa_x + aa_width, aa_y + aa_height), (255, 255, 255), thickness=-1)

    # Action Angle text
    action_text = f"Action Angle: {action_angle}"
    cv2.putText(frame, action_text, (aa_x + 10, aa_y + 25), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    return frame

def stream_video(rtmp_url, window_name):
    """
    Opens an RTMP stream and overlays the scoreboard in the top-left corner.
    """
    cap = cv2.VideoCapture(rtmp_url)

    if not cap.isOpened():
        print(f"Error: Unable to open the RTMP stream for {window_name}")
        return

    # Set resolution to Full HD
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Extract "GoPro_SU#" from the RTMP URL
    action_label = rtmp_url.split('/')[-1]  # Example: "GoPro_SU1"

    print(f"Streaming from {window_name} with overlay... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Unable to grab frame from {window_name}")
            break

        # Draw scoreboard overlay
        overlay_frame = draw_scoreboard(
            frame,
            tmobile_logo=tmobile_logo,
            seattle_logo=seattle_logo,
            home_logo=home_logo,
            away_logo=away_logo,
            home_acronym="SU",
            away_acronym="SU",
            home_score=0,  # Placeholder
            away_score=0,  # Placeholder
            action_angle=action_label
        )

        # Show the overlay frame
        cv2.imshow(window_name, overlay_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(window_name)

# Start threads for each RTMP stream
threads = []
for i, url in enumerate(rtmp_urls):
    thread = threading.Thread(target=stream_video, args=(url, f"GoPro {i+1}"))
    thread.start()
    threads.append(thread)

# Wait for all threads to finish
for thread in threads:
    thread.join()

cv2.destroyAllWindows()