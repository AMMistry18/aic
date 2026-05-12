"""Visual check for SC pose labels (YOLO pose, 4 kpts).

Navigation
----------
- **Next image:** left-click, or almost any keyboard key (exceptions below).
- **Quit:** ``q``.
- **Zoom:** mouse wheel on the image, or ``+`` / ``=`` zoom in and ``-`` zoom out,
  ``r`` reset zoom; Matplotlib toolbar also has pan/zoom if enabled.

YOLO pose visibility on circles
-------------------------------
Markers follow Ultralytics/YOLO-pose conventions used in labels:

  * **Filled disk** → visibility ``2``: in-frame and counted as strongly visible.
  * **Hollow ring** → visibility ``1``: in front of camera but clipped/off-image
    in the strict sense, or (in the collector script) optionally downgraded for
    heuristics — still plotted so you see where that keypoint landed.

Edges between corners are drawn only when **both** endpoints have visibility ``> 0``.
"""

from __future__ import annotations

import cv2
import glob
import os
import sys

# Default matches DataCollectorPoseSC
DATA_DIR = os.path.expanduser(
    os.environ.get("AIC_SC_COLLECT_OUTPUT_DIR", "~/aic_perception_data/pose_sc")
)
SPLIT = "train"  # change to "val" to check validation set

CLASS_NAMES = ["sc_port"]
NUM_KEYPOINTS = 4
MIN_PARTS = 1 + 4 + NUM_KEYPOINTS * 3

DET_COLORS = [
    (0, 220, 0),
    (0, 165, 255),
    (180, 105, 255),
    (255, 200, 0),
    (255, 0, 180),
    (200, 200, 0),
]

QUAD_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0)]
QUAD_EDGE_COLOR = (255, 180, 0)

DISPLAY_MAX_HEIGHT = 600
DISPLAY_MAX_WIDTH = 1060

# Interactive zoom clamps (applied to display-sized image pixels)
_ZOOM_MIN = 0.25
_ZOOM_MAX = 12.0


def _opencv_gui_available() -> bool:
    try:
        cv2.namedWindow("__check_sc_probe", cv2.WINDOW_AUTOSIZE)
        cv2.destroyWindow("__check_sc_probe")
        return True
    except cv2.error:
        return False


def _annotate_frame(img_path: str) -> tuple[cv2.typing.MatLike | None, str | None]:
    """Load image + labels and draw overlays. Returns (bgr,) or (None,) if skipped."""
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    h, w = img.shape[:2]

    filename = os.path.basename(img_path).replace(".png", ".txt")
    lbl_path = os.path.join(DATA_DIR, "labels", SPLIT, filename)
    if not os.path.exists(lbl_path):
        return None, None

    with open(lbl_path, encoding="utf-8") as f:
        raw_lines = [ln.strip() for ln in f if ln.strip()]

    for det_idx, line in enumerate(raw_lines):
        parts = [float(x) for x in line.split()]
        if len(parts) < MIN_PARTS:
            continue

        class_id = int(parts[0])
        box_color = DET_COLORS[det_idx % len(DET_COLORS)]

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
        tag = f"{name}#{det_idx}" if len(raw_lines) > 1 else name
        cv2.putText(
            img,
            tag,
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

        for a, b in QUAD_EDGES:
            va, vb = kp_xyv[a][2], kp_xyv[b][2]
            if va > 0 and vb > 0:
                cv2.line(
                    img,
                    (kp_xyv[a][0], kp_xyv[a][1]),
                    (kp_xyv[b][0], kp_xyv[b][1]),
                    QUAD_EDGE_COLOR,
                    1,
                    cv2.LINE_AA,
                )

        for i, (px, py, vis) in enumerate(kp_xyv):
            if vis > 0:
                # vis==2 filled, vis==1 outline (YOLO convention; matches collector flags)
                thickness = -1 if vis == 2 else 2
                cv2.circle(img, (px, py), 5, box_color, thickness)
                cv2.putText(
                    img,
                    str(i),
                    (px + 5, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    box_color,
                    1,
                    cv2.LINE_AA,
                )

    banner = os.path.basename(img_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.55, 2
    (tw, th), bl = cv2.getTextSize(banner, font, scale, thickness)
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
        banner,
        (pad_x, th + pad_y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    return img, banner


def _resize_for_display(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    h, w = img.shape[:2]
    disp_scale = min(
        1.0,
        DISPLAY_MAX_HEIGHT / h,
        DISPLAY_MAX_WIDTH / w,
    )
    if disp_scale < 1.0:
        return cv2.resize(
            img,
            (int(w * disp_scale), int(h * disp_scale)),
            interpolation=cv2.INTER_AREA,
        )
    return img


def _show_cv2(disp_bgr: cv2.typing.MatLike) -> bool:
    """Returns True if user wants to quit the whole slideshow.

    Zoom: mouse wheel, ``+``/``=`` in, ``-`` out, ``r`` reset. Next: click elsewhere.
    """
    win = "SC data verification"
    base = disp_bgr
    click_next = {"v": False}
    zoom = {"z": 1.0}
    try:
        _wheel_evt = cv2.EVENT_MOUSEWHEEL
    except AttributeError:
        _wheel_evt = None

    def redraw() -> None:
        z = max(_ZOOM_MIN, min(_ZOOM_MAX, float(zoom["z"])))
        zoom["z"] = z
        h, w = base.shape[:2]
        nw = max(1, int(round(w * z)))
        nh = max(1, int(round(h * z)))
        interp = cv2.INTER_LINEAR if z >= 1.0 else cv2.INTER_AREA
        out = cv2.resize(base, (nw, nh), interpolation=interp)
        cv2.imshow(win, out)

    def _on_mouse(ev, _x, _y, flags, *_args) -> None:
        if ev == cv2.EVENT_LBUTTONDOWN:
            click_next["v"] = True
        elif _wheel_evt is not None and ev == _wheel_evt:
            wd = flags >> 16
            wd = wd if wd < 32768 else wd - 65536  # signed 16-bit wheel delta (Win/Linux)
            if wd > 0:
                zoom["z"] *= 1.12
            elif wd < 0:
                zoom["z"] /= 1.12
            redraw()

    redraw()
    cv2.setMouseCallback(win, _on_mouse)
    while True:
        k = cv2.waitKey(30)
        if k != -1:
            kk = k & 0xFF
            if kk in (ord("q"), ord("Q")):
                return True
            if kk in (ord("="), ord("+")):
                zoom["z"] *= 1.15
                redraw()
                continue
            if kk in (ord("-"), ord("_")):
                zoom["z"] /= 1.15
                redraw()
                continue
            if kk in (ord("r"), ord("R")):
                zoom["z"] = 1.0
                redraw()
                continue
            return False
        if click_next["v"]:
            return False


def _mpl_axes_zoom_about(ax, xdata: float, ydata: float, zoom_in: bool) -> None:
    """Zoom imshow axes about (xdata, ydata); works with inverted image y-axis."""
    factor = 1.0 / 1.25 if zoom_in else 1.25
    x0, x1 = ax.get_xlim()
    ax.set_xlim(xdata + (x0 - xdata) * factor, xdata + (x1 - xdata) * factor)
    y0, y1 = ax.get_ylim()
    ax.set_ylim(ydata + (y0 - ydata) * factor, ydata + (y1 - ydata) * factor)


def _matplotlib_prepare_backend() -> bool:
    """Select backend before importing pyplot. Returns False if matplotlib missing."""
    try:
        import matplotlib
    except ImportError:
        return False
    mpl_backend = os.environ.get("MPLBACKEND", "TkAgg").strip() or "TkAgg"
    try:
        matplotlib.use(mpl_backend)
    except Exception:
        matplotlib.use("TkAgg")
    return True


def _show_matplotlib(rgb: cv2.typing.MatLike) -> bool:
    """Quit True on ``q``. Next: click or keys other than +-r. Zoom: wheel / ``+/-`` / ``r``."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is required when OpenCV has no GUI. "
            "`pixi` should ship it — try `pixi install`.",
            file=sys.stderr,
        )
        return True

    quit_all = {"v": False}
    advance = {"v": False}  # next image; must not set quit on close_event

    fig, ax = plt.subplots(figsize=(10, 8))
    mgr = getattr(fig.canvas, "manager", None)
    if mgr is not None and hasattr(mgr, "set_window_title"):
        mgr.set_window_title(
            "SC labels — wheel / +- zoom, r reset | click | key→next | q quit"
        )
    ax.imshow(rgb)
    ax.axis("off")
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    base_xlim = tuple(ax.get_xlim())
    base_ylim = tuple(ax.get_ylim())

    def on_key(ev) -> None:
        k = ev.key
        if k is None:
            return
        if k in ("q", "Q"):
            quit_all["v"] = True
            plt.close(fig)
            return
        if k in ("+", "plus", "="):
            xc = 0.5 * (ax.get_xlim()[0] + ax.get_xlim()[1])
            yc = 0.5 * (ax.get_ylim()[0] + ax.get_ylim()[1])
            _mpl_axes_zoom_about(ax, xc, yc, zoom_in=True)
            fig.canvas.draw_idle()
            return
        if k in ("-", "minus", "_"):
            xc = 0.5 * (ax.get_xlim()[0] + ax.get_xlim()[1])
            yc = 0.5 * (ax.get_ylim()[0] + ax.get_ylim()[1])
            _mpl_axes_zoom_about(ax, xc, yc, zoom_in=False)
            fig.canvas.draw_idle()
            return
        if k in ("r", "R"):
            ax.set_xlim(base_xlim)
            ax.set_ylim(base_ylim)
            fig.canvas.draw_idle()
            return
        advance["v"] = True
        plt.close(fig)

    def on_scroll(ev) -> None:
        if ev.inaxes != ax or ev.xdata is None or ev.ydata is None:
            return
        step = getattr(ev, "step", None)
        if step is None:
            zoom_in = getattr(ev, "button", "") == "up"
        else:
            zoom_in = step > 0
        _mpl_axes_zoom_about(ax, ev.xdata, ev.ydata, zoom_in=zoom_in)
        fig.canvas.draw_idle()

    def on_click(ev) -> None:
        if ev.button == 1:
            advance["v"] = True
            plt.close(fig)

    def on_close(_ev) -> None:
        # Closing after click/key fires this too — only WM "X" is a full quit.
        if quit_all["v"] or advance["v"]:
            return
        quit_all["v"] = True

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("close_event", on_close)
    plt.show(block=True)
    plt.close(fig)
    return quit_all["v"]


def main() -> None:
    img_paths = sorted(glob.glob(os.path.join(DATA_DIR, "images", SPLIT, "*.png")))
    preview_out = os.environ.get("CHECK_SC_OUT", "").strip()

    use_cv2 = _opencv_gui_available()
    if not use_cv2 and not _matplotlib_prepare_backend():
        print(
            "matplotlib is required when OpenCV has no GUI — run `pixi install`.",
            file=sys.stderr,
        )
        return

    backend = "OpenCV windows" if use_cv2 else f"matplotlib ({os.environ.get('MPLBACKEND', 'TkAgg')})"

    if not img_paths:
        print(
            f"No images in {os.path.join(DATA_DIR, 'images', SPLIT)}",
            file=sys.stderr,
        )
        return

    print(
        f"{len(img_paths)} images — {backend}. "
        "Next: click or most keys. Quit: q. Zoom: mouse wheel or +/- keys, r resets."
        + (f" CHECK_SC_OUT={preview_out}" if preview_out else ""),
        file=sys.stderr,
    )

    if not use_cv2 and not os.environ.get("DISPLAY"):
        print(
            "DISPLAY is unset; GUI may fail under SSH without X forwarding. "
            "Use ssh -Y / WSL or set CHECK_SC_OUT=/path.",
            file=sys.stderr,
        )

    for img_path in img_paths:
        img_bgr, _ = _annotate_frame(img_path)
        if img_bgr is None:
            continue
        disp_bgr = _resize_for_display(img_bgr)

        if preview_out:
            os.makedirs(preview_out, exist_ok=True)
            out_path = os.path.join(preview_out, os.path.basename(img_path))
            cv2.imwrite(out_path, disp_bgr)

        if use_cv2:
            if _show_cv2(disp_bgr):
                break
            continue

        rgb = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)
        if _show_matplotlib(rgb):
            break

    if use_cv2:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
