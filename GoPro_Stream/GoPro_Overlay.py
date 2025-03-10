import cv2
import threading
import numpy as np

###############################################################################
# RTMP URLs
###############################################################################
rtmp_urls = [
    "rtmp://192.168.1.100/live/GoPro_SU1",
    "rtmp://192.168.1.100/live/GoPro_SU2",
    "rtmp://192.168.1.100/live/GoPro_SU3",
    "rtmp://192.168.1.100/live/GoPro_SU4"
]

###############################################################################
# Load Logos (Adjust Paths)
###############################################################################
tmobile_logo = cv2.imread("assets/TMobile_Logo.png")
seattle_logo = cv2.imread("assets/SeattleU_SponsorLogo.png")
home_logo    = cv2.imread("assets/SeattleU_Logo.png")
away_logo    = cv2.imread("assets/away_team.png")

###############################################################################
# Helper Functions for Alpha Blending & Drawing
###############################################################################
def draw_transparent_rect(frame, x, y, w, h, color, alpha=0.5):
    """
    Draws a semi-transparent rectangle on 'frame' at (x,y) of size (w,h).
    'color' is a BGR tuple, 'alpha' is from 0.0 (fully transparent) to 1.0 (opaque).
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness=-1)
    # Blend the overlay with the original frame
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_image(frame, image, x, y, width, height):
    """
    Places 'image' (BGR) onto 'frame' at (x,y) after resizing to (width,height).
    Simple overwrite (no alpha channel).
    """
    if image is None:
        return
    resized = cv2.resize(image, (width, height))
    h, w = frame.shape[:2]
    if x >= w or y >= h:
        return  # Off-screen, skip
    end_x = min(x + width, w)
    end_y = min(y + height, h)
    roi_width = end_x - x
    roi_height = end_y - y
    frame[y:end_y, x:end_x] = resized[:roi_height, :roi_width]

def draw_white_box_with_text(
    frame, text, x, y, box_width, box_height,
    font, font_scale, thickness, text_color=(0,0,0), alpha=0.7
):
    """
    Draws a semi-transparent white rectangle (box) and centers 'text' inside it.
    """
    # Semi-transparent white background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + box_width, y + box_height), (255,255,255), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Calculate text size
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    # Center the text inside the box
    cx = x + (box_width - text_w) // 2
    cy = y + (box_height + text_h) // 2 - 2

    cv2.putText(frame, text, (cx, cy), font, font_scale, text_color, thickness, cv2.LINE_AA)

###############################################################################
# Main Overlay Function
###############################################################################
def draw_custom_overlay(
    frame,
    tmobile_logo,
    seattle_logo,
    home_logo,
    away_logo,
    home_acronym="HOM",
    away_acronym="AWY",
    home_score=0,
    away_score=0,
    action_angle="GoPro_SU1"
):
    """
    Draws a scoreboard in the top-left:
      - T-Mobile (80×80) + SeattleU (80×80) side by side
      - Home row: 40×40 cells for (logo, acronym, score)
      - Away row: 40×40 cells for (logo, acronym, score)
    Then in the top-right:
      - Red box labeled "Action Angle"
      - White box next to it with the GoPro name
    """

    # Choose a refined font
    font = cv2.FONT_HERSHEY_DUPLEX

    # Scoreboard overall bounding box (semi-transparent background)
    sb_x, sb_y = 10, 10
    sb_width = 280  # 80+80 for sponsor logos + 120 for home/away cells
    sb_height = 80  # 80 high total (2 rows of 40, but T-Mobile/SeattleU are 80 high)
    draw_transparent_rect(frame, sb_x, sb_y, sb_width, sb_height, (0,0,0), alpha=0.2)

    ########################################################################
    # 1) T-Mobile & SeattleU side by side, each 80×80
    ########################################################################
    # T-Mobile
    draw_image(frame, tmobile_logo, sb_x, sb_y, 80, 80)
    # SeattleU (just to the right of T-Mobile)
    draw_image(frame, seattle_logo, sb_x + 80, sb_y, 80, 80)

    ########################################################################
    # 2) Home row (40 high) to the right, top half of the scoreboard
    ########################################################################
    # We'll place it at x= 10 + 160 = 170, y= 10, each cell 40×40
    row1_x = sb_x + 160
    row1_y = sb_y
    cell_size = 40

    # Home logo
    draw_image(frame, home_logo, row1_x, row1_y, cell_size, cell_size)
    # Home acronym
    draw_white_box_with_text(
        frame, home_acronym,
        row1_x + cell_size, row1_y,
        cell_size, cell_size,
        font, font_scale=0.7, thickness=2
    )
    # Home score
    draw_white_box_with_text(
        frame, str(home_score),
        row1_x + (cell_size * 2), row1_y,
        cell_size, cell_size,
        font, font_scale=0.7, thickness=2
    )

    ########################################################################
    # 3) Away row (40 high) below the home row
    ########################################################################
    row2_x = sb_x + 160
    row2_y = sb_y + 40  # second row is 40px below the first row
    # Away logo
    draw_image(frame, away_logo, row2_x, row2_y, cell_size, cell_size)
    # Away acronym
    draw_white_box_with_text(
        frame, away_acronym,
        row2_x + cell_size, row2_y,
        cell_size, cell_size,
        font, font_scale=0.7, thickness=2
    )
    # Away score
    draw_white_box_with_text(
        frame, str(away_score),
        row2_x + (cell_size * 2), row2_y,
        cell_size, cell_size,
        font, font_scale=0.7, thickness=2
    )

    ########################################################################
    # 4) Action Angle in top-right
    ########################################################################
    frame_h, frame_w = frame.shape[:2]

    # Each box is 120 wide, 40 high
    box_w, box_h = 120, 40
    spacing = 5
    total_w = (box_w * 2) + spacing  # red box + white box + spacing

    aa_x = frame_w - total_w - 10
    aa_y = 10

    # Red box for "Action Angle"
    draw_transparent_rect(frame, aa_x, aa_y, box_w, box_h, (0,0,255), alpha=0.7)
    # Center "Action Angle" text
    text = "Action Angle"
    text_size, _ = cv2.getTextSize(text, font, 0.7, 2)
    tw, th = text_size
    cx = aa_x + (box_w - tw)//2
    cy = aa_y + (box_h + th)//2 - 2
    cv2.putText(frame, text, (cx, cy), font, 0.7, (0,0,0), 2, cv2.LINE_AA)

    # White box for the GoPro name
    box2_x = aa_x + box_w + spacing
    draw_transparent_rect(frame, box2_x, aa_y, box_w, box_h, (255,255,255), alpha=0.7)
    # Center the action_angle text
    text = action_angle
    text_size, _ = cv2.getTextSize(text, font, 0.7, 2)
    tw, th = text_size
    cx = box2_x + (box_w - tw)//2
    cy = aa_y + (box_h + th)//2 - 2
    cv2.putText(frame, text, (cx, cy), font, 0.7, (0,0,0), 2, cv2.LINE_AA)

    return frame

###############################################################################
# Threaded RTMP Retrieval
###############################################################################
def stream_video(rtmp_url, window_name):
    cap = cv2.VideoCapture(rtmp_url)
    if not cap.isOpened():
        print(f"Error: Unable to open the RTMP stream for {window_name}")
        return

    # Optional: set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Extract "GoPro_SU#" from the URL
    action_label = rtmp_url.split('/')[-1]

    print(f"Streaming from {window_name} with custom overlay... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Unable to grab frame from {window_name}")
            break

        # Draw custom overlay
        overlayed_frame = draw_custom_overlay(
            frame,
            tmobile_logo=tmobile_logo,
            seattle_logo=seattle_logo,
            home_logo=home_logo,
            away_logo=away_logo,
            home_acronym="SU",   # Example
            away_acronym="UW",   # Example
            home_score=0,        # Example
            away_score=0,        # Example
            action_angle=action_label
        )

        cv2.imshow(window_name, overlayed_frame)

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

for thread in threads:
    thread.join()

cv2.destroyAllWindows()