import cv2
import threading
import numpy as np
from PIL import Image, ImageDraw, ImageFont

###############################################################################
# RTMP URLs
###############################################################################
rtmp_urls = [
    #"rtmp://192.168.1.100/live/GoPro_SU1",
    #"rtmp://192.168.1.100/live/GoPro_SU2",
    "rtmp://192.168.1.100/live/GoPro_SU3",
    #"rtmp://192.168.1.100/live/GoPro_SU4"
]

###############################################################################
# Load Logos (adjust paths)
###############################################################################
tmobile_logo = cv2.imread("assets/Tmobile_Logo.png")
seattle_logo = cv2.imread("assets/SeattleU_SponsorLogo.png")
home_logo    = cv2.imread("assets/SeattleU_Logo.png")
away_logo    = cv2.imread("assets/UW_Logo.png")

###############################################################################
# Load Custom Fonts
###############################################################################
FONT_PATH = "assets/FuturaMaxi.otf"  # Example: Main scoreboard font
ANGLE_FONT_PATH = "assets/FuturaMaxi.otf"  # Example: Smaller font for action angle
FONT_SIZE = 16  # Scoreboard font size
ANGLE_FONT_SIZE = 12  # Smaller font for the Action Angle row

def load_font(size, font_path):
    return ImageFont.truetype(font_path, size)

###############################################################################
# Helper Functions
###############################################################################
def draw_pil_text(img, text, x, y, w, h, font, text_color=(0, 0, 0)):
    """
    Draw text **fully centered** in a box using PIL (fixes vertical alignment).
    """
    # Convert OpenCV image to PIL
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Measure text size correctly
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]  # Width of text
    text_h = bbox[3] - bbox[1]  # Height of text

    # **Fix Vertical Centering:** Adjust Y based on text height
    centered_x = x + (w - text_w) // 2  # Center horizontally
    centered_y = y + (h - text_h) // 2 + 2  # Center vertically with slight adjustment

    # ✅ Draw centered text (No bold effect)
    draw.text((centered_x, centered_y), text, font=font, fill=text_color)

    # Convert back to OpenCV
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_image(frame, image, x, y, width, height):
    """
    Places 'image' onto 'frame' at (x, y) after resizing to (width, height).
    """
    if image is None:
        return
    resized = cv2.resize(image, (width, height))
    frame[y:y+height, x:x+width] = resized

def draw_acronym_score_box(frame, acronym, score, x, y, w, h, font):
    """
    Draws a **pure white** box with a black vertical line separating acronym & score.
    Uses PIL for **perfectly centered text**.
    """
    # ✅ Solid white background
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), thickness=-1)

    # ✅ Black vertical separator line
    mid_x = x + w // 2
    cv2.line(frame, (mid_x, y), (mid_x, y + h), (0, 0, 0), 2)

    # ✅ Draw **centered** text
    frame = draw_pil_text(frame, acronym, x, y, w // 2, h, font, text_color=(0, 0, 0))  # Left side
    frame = draw_pil_text(frame, score, mid_x, y, w // 2, h, font, text_color=(0, 0, 0))  # Right side

    return frame

###############################################################################
# Main Overlay Function
###############################################################################
def draw_custom_overlay(frame, home_acronym="HOM", away_acronym="AWY",
                        home_score="0", away_score="0", action_angle="GoPro_SU1"):
    """
    Draws scoreboard overlay in the top-left and "CS 25.15 | Action Angle: GoProName" below it.
    Uses custom fonts for acronyms, scores, and labels.
    """
    font = load_font(FONT_SIZE, FONT_PATH)  # Load main scoreboard font
    angle_font = load_font(ANGLE_FONT_SIZE, ANGLE_FONT_PATH)  # Load smaller font for Action Angle row

    # 1) T-Mobile & SeattleU logos
    scoreboard_x, scoreboard_y = 10, 10
    draw_image(frame, tmobile_logo, scoreboard_x, scoreboard_y, 80, 80)
    draw_image(frame, seattle_logo, scoreboard_x + 80, scoreboard_y, 80, 80)

    # 2) Home row (40×40)
    home_row_x = scoreboard_x + 160
    home_row_y = scoreboard_y
    draw_image(frame, home_logo, home_row_x, home_row_y, 40, 40)
    frame = draw_acronym_score_box(frame, home_acronym, home_score, home_row_x + 40, home_row_y, 80, 40, font)

    # 3) Away row (40×40)
    away_row_x = scoreboard_x + 160
    away_row_y = scoreboard_y + 40
    draw_image(frame, away_logo, away_row_x, away_row_y, 40, 40)
    frame = draw_acronym_score_box(frame, away_acronym, away_score, away_row_x + 40, away_row_y, 80, 40, font)

    # 4) ✅ New row directly below the scoreboard (No Space)
    info_box_y = away_row_y + 40  # Directly below away row (40px tall)
    info_box_w = 280  # Width same as the scoreboard
    info_box_h = 30  # Height for compact text

    # ✅ Solid white background with black text
    cv2.rectangle(frame, (scoreboard_x, info_box_y), (scoreboard_x + info_box_w, info_box_y + info_box_h), (255, 255, 255), thickness=-1)

    # ✅ Draw the text **perfectly centered** inside the box
    info_text = f"CS 25.15  |  ACTION ANGLE: {action_angle}"
    frame = draw_pil_text(frame, info_text, scoreboard_x, info_box_y, info_box_w, info_box_h, angle_font, text_color=(0, 0, 0))
    return frame

###############################################################################
# RTMP Streaming with Overlay
###############################################################################
def stream_video(rtmp_url, window_name):
    cap = cv2.VideoCapture(rtmp_url)
    if not cap.isOpened():
        print(f"Error: Unable to open the RTMP stream for {window_name}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    action_label = rtmp_url.split('/')[-1]

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Unable to grab frame from {window_name}")
            break

        overlayed_frame = draw_custom_overlay(
            frame, home_acronym="SU", away_acronym="UW",
            home_score="0", away_score="0", action_angle=action_label
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