"""
prepare_images.py -- FreshTrack CV assignment image preparation.

Downloads individual fruit images from the Fruits-360 dataset (GitHub mirror
of https://www.kaggle.com/datasets/moltean/fruits), downloads the classic
OpenCV `j.png` morphology demo image, and programmatically generates the
chessboard, binary-morphology demo, and the `mixed_bowl.jpg` batch image
(composited from Fruits-360 fruits so that the ground-truth count of round
fruits is known exactly).

Run once before executing the notebook:
    python prepare_images.py
"""

import os
import urllib.request

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(HERE, "images")

FRUITS_360_BASE = (
    "https://raw.githubusercontent.com/Horea94/Fruit-Images-Dataset/master/Training"
)

# fruit file name -> Fruits-360 class folder / image
DOWNLOADS = {
    "red_apple.jpg": "Apple%20Red%201/0_100.jpg",
    "green_apple.jpg": "Apple%20Granny%20Smith/0_100.jpg",
    "banana.jpg": "Banana/0_100.jpg",
    "strawberry.jpg": "Strawberry/0_100.jpg",
    "orange.jpg": "Orange/0_100.jpg",
    "lime.jpg": "Limes/0_100.jpg",
}

J_PNG_URL = "https://docs.opencv.org/4.x/j.png"

# Ground truth for the composited mixed bowl:
#   5 apples (4 red + 1 green) + 3 oranges + 1 lime = 9 round fruits
#   2 bananas (elongated, must NOT be counted by the circle detector)
ROUND_FRUITS_GROUND_TRUTH = 9


def _download(url: str, dst: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dst, "wb") as fh:
        fh.write(resp.read())


def download_fruits() -> None:
    for name, rel in DOWNLOADS.items():
        dst = os.path.join(IMG_DIR, name)
        if os.path.exists(dst):
            continue
        _download(f"{FRUITS_360_BASE}/{rel}", dst)
        # upscale the 100x100 originals to 300x300 for nicer plots/histograms
        img = cv2.imread(dst)
        img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(dst, img)
        print(f"downloaded {name}")


def download_j() -> None:
    dst = os.path.join(IMG_DIR, "j.png")
    if os.path.exists(dst):
        return
    try:
        _download(J_PNG_URL, dst)
        img = cv2.imread(dst, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError("bad download")
        print("downloaded j.png")
    except Exception:
        # fallback: draw a 'j' ourselves
        img = np.zeros((250, 200), np.uint8)
        cv2.putText(img, "j", (40, 190), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 7, 255, 12)
        cv2.imwrite(dst, img)
        print("generated fallback j.png")


def make_dark_banana() -> None:
    """Simulated under-exposed warehouse shot of the banana."""
    dst = os.path.join(IMG_DIR, "banana_dark.jpg")
    if os.path.exists(dst):
        return
    banana = cv2.imread(os.path.join(IMG_DIR, "banana.jpg"))
    dark = np.clip(banana.astype(np.float32) * 0.55 - 25, 0, 255).astype(np.uint8)
    cv2.imwrite(dst, dark)
    print("generated banana_dark.jpg")


def make_chessboard() -> None:
    dst = os.path.join(IMG_DIR, "chessboard.png")
    if os.path.exists(dst):
        return
    sq, n, border = 55, 8, 16
    board = np.zeros((n * sq, n * sq), np.uint8)
    for r in range(n):
        for c in range(n):
            if (r + c) % 2 == 0:
                board[r * sq:(r + 1) * sq, c * sq:(c + 1) * sq] = 255
    board = cv2.copyMakeBorder(board, border, border, border, border,
                               cv2.BORDER_CONSTANT, value=0)
    cv2.imwrite(dst, board)
    print("generated chessboard.png")


def make_morphology_demo() -> None:
    """Binary demo image: text + shapes, with salt noise outside and
    pepper holes inside -- the classic morphology playground."""
    dst = os.path.join(IMG_DIR, "morphology.png")
    if os.path.exists(dst):
        return
    rng = np.random.default_rng(42)
    img = np.zeros((300, 600), np.uint8)
    cv2.putText(img, "FRESHTRACK", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 255, 10)
    cv2.circle(img, (120, 220), 45, 255, -1)
    cv2.rectangle(img, (250, 180), (400, 260), 255, -1)
    cv2.ellipse(img, (500, 220), (60, 35), 20, 0, 360, 255, -1)
    # salt noise (white specks on background)
    ys, xs = rng.integers(0, 300, 400), rng.integers(0, 600, 400)
    for y, x in zip(ys, xs):
        cv2.circle(img, (int(x), int(y)), int(rng.integers(1, 3)), 255, -1)
    # pepper holes (black holes inside shapes)
    ys, xs = rng.integers(0, 300, 250), rng.integers(0, 600, 250)
    for y, x in zip(ys, xs):
        cv2.circle(img, (int(x), int(y)), int(rng.integers(1, 4)), 0, -1)
    cv2.imwrite(dst, img)
    print("generated morphology.png")


def _fruit_mask(fruit_bgr: np.ndarray) -> np.ndarray:
    """Foreground mask for a Fruits-360 image (white background)."""
    hsv = cv2.cvtColor(fruit_bgr, cv2.COLOR_BGR2HSV)
    not_white = ((hsv[:, :, 1] > 25) | (hsv[:, :, 2] < 210)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    not_white = cv2.morphologyEx(not_white, cv2.MORPH_CLOSE, kernel)
    not_white = cv2.morphologyEx(not_white, cv2.MORPH_OPEN, kernel)
    return not_white


def _paste(canvas: np.ndarray, fruit: np.ndarray, center: tuple, size: int,
           angle: float = 0.0) -> None:
    """Alpha-blend a Fruits-360 fruit onto the canvas with a soft shadow."""
    fruit = cv2.resize(fruit, (size, size), interpolation=cv2.INTER_CUBIC)
    if angle:
        m = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
        fruit = cv2.warpAffine(fruit, m, (size, size),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(255, 255, 255))
    mask = _fruit_mask(fruit)
    alpha = cv2.GaussianBlur(mask, (7, 7), 0).astype(np.float32) / 255.0

    cx, cy = center
    x0, y0 = cx - size // 2, cy - size // 2
    x1, y1 = x0 + size, y0 + size
    if x0 < 0 or y0 < 0 or x1 > canvas.shape[1] or y1 > canvas.shape[0]:
        raise ValueError(f"fruit at {center} size {size} leaves the canvas")

    # soft drop shadow, offset down-right
    sh = np.zeros(canvas.shape[:2], np.float32)
    sh_y0, sh_x0 = y0 + 12, x0 + 10
    sh[sh_y0:sh_y0 + size, sh_x0:sh_x0 + size] = alpha[
        : canvas.shape[0] - sh_y0, : canvas.shape[1] - sh_x0]
    sh = cv2.GaussianBlur(sh, (31, 31), 0) * 0.35
    canvas[:] = (canvas.astype(np.float32) * (1 - sh[..., None])).astype(np.uint8)

    roi = canvas[y0:y1, x0:x1].astype(np.float32)
    blended = roi * (1 - alpha[..., None]) + fruit.astype(np.float32) * alpha[..., None]
    canvas[y0:y1, x0:x1] = blended.astype(np.uint8)


def make_mixed_bowl() -> None:
    dst = os.path.join(IMG_DIR, "mixed_bowl.jpg")
    if os.path.exists(dst):
        return
    rng = np.random.default_rng(7)

    # light countertop background with a soft vertical gradient + grain
    h, w = 700, 900
    grad = np.linspace(235, 205, h, dtype=np.float32)[:, None]
    canvas = np.repeat(grad, w, axis=1)
    canvas = np.stack([canvas - 4, canvas, canvas + 2], axis=-1)  # slightly warm
    canvas += rng.normal(0, 2.5, canvas.shape).astype(np.float32)
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)

    red = cv2.imread(os.path.join(IMG_DIR, "red_apple.jpg"))
    green = cv2.imread(os.path.join(IMG_DIR, "green_apple.jpg"))
    orange = cv2.imread(os.path.join(IMG_DIR, "orange.jpg"))
    lime = cv2.imread(os.path.join(IMG_DIR, "lime.jpg"))
    banana = cv2.imread(os.path.join(IMG_DIR, "banana.jpg"))

    # --- round fruits: 5 apples + 3 oranges + 1 lime = 9 (ground truth) ---
    _paste(canvas, red, (150, 180), 150, angle=10)
    _paste(canvas, red, (360, 170), 140, angle=-25)
    _paste(canvas, red, (150, 410), 156, angle=40)
    _paste(canvas, red, (370, 405), 144, angle=95)
    _paste(canvas, green, (150, 605), 140, angle=-15)
    _paste(canvas, orange, (600, 170), 160, angle=0)
    _paste(canvas, orange, (790, 320), 150, angle=30)
    _paste(canvas, orange, (615, 430), 156, angle=-40)
    _paste(canvas, lime, (815, 140), 95, angle=20)
    # --- elongated fruits: 2 bananas (must not be counted as circles) ---
    _paste(canvas, banana, (560, 610), 175, angle=15)
    _paste(canvas, banana, (770, 590), 175, angle=-30)

    # mild sensor noise so the pipeline has something realistic to clean
    noisy = np.clip(
        canvas.astype(np.float32) + rng.normal(0, 4, canvas.shape), 0, 255
    ).astype(np.uint8)
    cv2.imwrite(dst, noisy, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"generated mixed_bowl.jpg (ground truth: {ROUND_FRUITS_GROUND_TRUTH} "
          "round fruits + 2 bananas)")


def ensure_images() -> str:
    os.makedirs(IMG_DIR, exist_ok=True)
    download_fruits()
    download_j()
    make_dark_banana()
    make_chessboard()
    make_morphology_demo()
    make_mixed_bowl()
    return IMG_DIR


if __name__ == "__main__":
    ensure_images()
    print("all images ready in", IMG_DIR)
