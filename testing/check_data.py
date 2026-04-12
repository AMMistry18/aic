import cv2
import os
import glob

# Point this to your new output folder
DATA_DIR = os.path.expanduser("~/aic_perception_data/pose")
SPLIT = "train"  # change to "val" to check validation set

# Matches DataCollectorPose2 / aic_ports_pose.yaml
CLASS_NAMES = [f"nic_card_{i}" for i in range(5)]
NUM_KEYPOINTS = 8
# YOLO-Pose line: class + cx,cy,w,h + (x,y,v) * NUM_KEYPOINTS
MIN_PARTS = 1 + 4 + NUM_KEYPOINTS * 3

# BGR — one color per mount slot class
CLASS_COLORS = [
    (0, 220, 0),      # nic_card_0 — green
    (0, 165, 255),    # nic_card_1 — orange
    (180, 105, 255),  # nic_card_2 — pink
    (255, 200, 0),    # nic_card_3 — light blue
    (255, 0, 180),    # nic_card_4 — purple-ish
]

# Port-0 quad (KP 0–3) vs port-1 quad (KP 4–7) — edge colors for skeleton
PORT0_EDGE_COLOR = (0, 0, 255)   # red
PORT1_EDGE_COLOR = (255, 128, 0) # cyan-tint

PORT0_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
PORT1_EDGES = [(4, 5), (5, 6), (6, 7), (7, 4)]

# Downscale for imshow only (labels are drawn at full resolution first).
DISPLAY_MAX_HEIGHT = 600
DISPLAY_MAX_WIDTH = 1060

img_paths = glob.glob(os.path.join(DATA_DIR, "images", SPLIT, "*.png"))

if not img_paths:
    print(f"No images found in {os.path.join(DATA_DIR, 'images', SPLIT)}. Did the data collector run?")

for img_path in img_paths:
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    # Find matching label file
    filename = os.path.basename(img_path).replace(".png", ".txt")
    lbl_path = os.path.join(DATA_DIR, "labels", SPLIT, filename)

    if not os.path.exists(lbl_path):
        continue

    with open(lbl_path, "r") as f:
        for line in f:
            parts = [float(x) for x in line.strip().split()]

            if len(parts) < MIN_PARTS:
                continue

            class_id = int(parts[0])
            box_color = CLASS_COLORS[class_id % len(CLASS_COLORS)]

            cx, cy, bw, bh = parts[1:5]
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)

            name = (
                CLASS_NAMES[class_id]
                if 0 <= class_id < len(CLASS_NAMES)
                else f"class_{class_id}"
            )
            cv2.putText(
                img,
                name,
                (x1, max(12, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2,
                cv2.LINE_AA,
            )

            kps = parts[5 : 5 + NUM_KEYPOINTS * 3]
            kp_xyv = []
            for i in range(NUM_KEYPOINTS):
                px = int(kps[i * 3] * w)
                py = int(kps[i * 3 + 1] * h)
                vis = int(kps[i * 3 + 2])
                kp_xyv.append((px, py, vis))

            def draw_quad(edges, color):
                for a, b in edges:
                    va, vb = kp_xyv[a][2], kp_xyv[b][2]
                    if va > 0 and vb > 0:
                        cv2.line(
                            img,
                            (kp_xyv[a][0], kp_xyv[a][1]),
                            (kp_xyv[b][0], kp_xyv[b][1]),
                            color,
                            1,
                            cv2.LINE_AA,
                        )

            draw_quad(PORT0_EDGES, PORT0_EDGE_COLOR)
            draw_quad(PORT1_EDGES, PORT1_EDGE_COLOR)

            for i, (px, py, vis) in enumerate(kp_xyv):
                if vis > 0:
                    thickness = -1 if vis == 2 else 2
                    cv2.circle(img, (px, py), 4, box_color, thickness)
                    cv2.putText(
                        img,
                        str(i),
                        (px + 5, py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        box_color,
                        1,
                        cv2.LINE_AA,
                    )

    # Filename banner at top (readable on any background)
    name = os.path.basename(img_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.55, 2
    (tw, th), bl = cv2.getTextSize(name, font, scale, thickness)
    pad_y, pad_x = 8, 10
    cv2.rectangle(
        img,
        (0, 0),
        (min(w, tw + 2 * pad_x), th + bl + 2 * pad_y),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        img,
        name,
        (pad_x, th + pad_y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    # Show the image (fit to screen). Press 'q' to quit, any other key for next.
    disp_scale = min(
        1.0,
        DISPLAY_MAX_HEIGHT / h,
        DISPLAY_MAX_WIDTH / w,
    )
    if disp_scale < 1.0:
        disp = cv2.resize(
            img,
            (int(w * disp_scale), int(h * disp_scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        disp = img
    cv2.imshow("Data Verification", disp)
    key = cv2.waitKey(0) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
