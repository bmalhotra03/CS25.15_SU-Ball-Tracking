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
# Load Custom Font (Change this to your desired font)
###############################################################################
FONT_PATH = "assets/FuturaMaxi.otf"  # Example: Use any TTF font
FONT_SIZE = 22  # Adjust for text size

def load_font(size):
    return ImageFont.truetype(FONT_PATH, size)

###############################################################################
# Helper Functions
###############################################################################
def draw_pil_text(img, text, x, y, font, text_color=(0, 0, 0)):
    """
    Draw text using Pillow (PIL) to support custom fonts, then overlay on OpenCV frame.
    """
    # Convert OpenCV image to PIL
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    # Draw text
    draw.text((x, y), text, font=font, fill=text_color)

    # Convert back to OpenCV
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def draw_transparent_rect(frame, x, y, w, h, color, alpha=0.5):
    """
    Draw a semi-transparent rectangle on 'frame' at (x, y) of size (w, h).
    'color' is BGR, 'alpha' in [0.0, 1.0].
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness=-1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

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
    Draws a white box with a vertical black line splitting acronym & score.
    Uses PIL for custom font rendering.
    """
    draw_transparent_rect(frame, x, y, w, h, (255, 255, 255), alpha=0.9)
    
    # Draw vertical black line in the middle
    mid_x = x + w // 2
    cv2.line(frame, (mid_x, y), (mid_x, y + h), (0, 0, 0), 2)

    # Draw text with PIL
    frame = draw_pil_text(frame, acronym, x + 10, y + 5, font, text_color=(0, 0, 0))  # Left side
    frame = draw_pil_text(frame, score, mid_x + 10, y + 5, font, text_color=(0, 0, 0))  # Right side
    return frame

###############################################################################
# Main Overlay Function
###############################################################################
def draw_custom_overlay(frame, home_acronym="HOM", away_acronym="AWY",
                        home_score="0", away_score="0", action_angle="GoPro_SU1"):
    """
    Draws scoreboard overlay in the top-left and "Action Angle" in the top-right.
    Uses custom fonts for acronyms, scores, and action angle text.
    """
    font = load_font(FONT_SIZE)  # Load custom font
    
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

    # 4) Action Angle (Top-Right)
    frame_h, frame_w = frame.shape[:2]
    box_w, box_h = 100, 40
    spacing = 5
    total_w = (box_w * 2) + spacing
    aa_x = frame_w - total_w - 10
    aa_y = 10

    draw_transparent_rect(frame, aa_x, aa_y, box_w, box_h, (0, 0, 255), alpha=0.7)
    draw_transparent_rect(frame, aa_x + box_w + spacing, aa_y, box_w, box_h, (255, 255, 255), alpha=0.7)

    # Draw action angle text with custom font
    frame = draw_pil_text(frame, "Action Angle", aa_x + 10, aa_y + 5, font, text_color=(0, 0, 0))
    frame = draw_pil_text(frame, action_angle, aa_x + box_w + spacing + 10, aa_y + 5, font, text_color=(0, 0, 0))

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
