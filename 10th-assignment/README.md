# Week 10 — FreshTrack Fruit Sorting: OpenCV Computer Vision Pipeline

Computer vision pipeline for a produce supply-chain scenario: warehouses receive mixed
crates of fruit and need to automatically **sort by type (color)**, **clean up noisy
camera images**, **detect shape features**, and **count individual round fruits** in a
batch image.

## Deliverable

**[`W10_FreshTrack_Fruit_CV_Pipeline.ipynb`](W10_FreshTrack_Fruit_CV_Pipeline.ipynb)** —
fully executed notebook with all outputs visible.

| Part | Contents |
|---|---|
| **A — Color & segmentation** | BGR vs RGB display, HSV channel analysis, `cv2.inRange` masking (red hue wrap-around), per-fruit HSV color table, brightness/contrast correction (NumPy vs `convertScaleAbs`), HSV-vs-RGB reflection + dark-fruit failure demo |
| **B — Histograms & filtering** | Grayscale histogram analysis, `equalizeHist` vs CLAHE (clip 2.0 / 8.0), Gaussian-noise denoising shoot-out (Gaussian / median / bilateral, PSNR), manual `conv2d()` vs `cv2.filter2D` with border-padding analysis |
| **C — Morphology** | Erosion / opening / closing on a noisy banana mask, morphological gradient outline overlay, morphology-on-BGR reflection, full morphology zoo on `j.png` (Top Hat isolates thin strokes) |
| **D — Shape & features** | Canny at two threshold pairs + pipeline explanation, `CannyEdgeDetector` from scratch (blur → gradients → NMS → double threshold → hysteresis), Harris corners on the chessboard with threshold sweep, `HoughCircles` fruit counting (**9/9 vs ground truth**), final end-to-end pipeline (load → HSV mask → morphology → Canny → bounding boxes → `safe_imwrite`) |

## Images

Fruit images are a subset of the [Fruits-360 dataset](https://www.kaggle.com/datasets/moltean/fruits)
(fetched from its GitHub mirror). `j.png` is the classic OpenCV morphology demo image.
`chessboard.png` and `morphology.png` are generated programmatically. `mixed_bowl.jpg` is
**composited from Fruits-360 fruits** by [`prepare_images.py`](prepare_images.py), so the
ground-truth count of round fruits is known exactly:
**5 apples + 3 oranges + 1 lime = 9 round fruits, plus 2 bananas** (which the circle
detector must not count).

All images land in `images/` on first run; the notebook's first cell calls
`ensure_images()` so a fresh clone regenerates everything automatically.

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace W10_FreshTrack_Fruit_CV_Pipeline.ipynb
# or just open it in Jupyter/VS Code and Run All
```

## Key results

- **Saturation (S)** channel separates fruit from the white background best (background S ≈ 0).
- Working HSV ranges per fruit recorded in the Part A color table (red needs two hue ranges, 0–15 and 170–180, because hue wraps).
- Brightness correction (`alpha=1.3, beta=20`) recovers the HSV mask on the under-exposed banana (coverage ~0% → ~37%).
- **CLAHE clip=2.0** helps segmentation most; global equalization amplifies background noise.
- **Bilateral filter** wins the denoising shoot-out — smooths noise while keeping the fruit edge sharp.
- Manual `conv2d` matches `cv2.filter2D` exactly (MAD = 0) once border padding matches (`BORDER_REFLECT_101`).
- **Top Hat** isolates the thin strokes of `j.png`.
- Canny [100, 200] captures the fruit outline cleanly; [30, 100] fires on interior texture.
- `HoughCircles` with `param1=100, param2=27, minRadius=30, maxRadius=100` counts **9/9 round fruits** and ignores both bananas.
- Final pipeline output saved to `outputs/orange_detection.jpg` (3/3 oranges boxed).

## Files

```
10th-assignment/
├── W10_FreshTrack_Fruit_CV_Pipeline.ipynb   # main deliverable (executed)
├── prepare_images.py                        # image download / generation
├── requirements.txt
├── images/                                  # created on first run
└── outputs/                                 # saved pipeline results
```
