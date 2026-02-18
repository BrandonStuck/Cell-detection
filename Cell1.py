import cv2
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from scipy.ndimage import gaussian_laplace
from scipy.ndimage import maximum_filter
from sklearn.cluster import DBSCAN

"""
To do:
    1) Refine 20x images
    2) Add option for 10x and 4x (doesnt have to be too reliable, just adjust parameters)
    3) Turn on voltage
    4) Prepare script to run on laptop (interface, folders)"""

MAG_CONFIGS = {
    "20x": {
        "radius_px": (10, 22),     # <-- tune once with one good image
        "peak_percentile": 99.6,
        "stripe_gate_dist": 15,
        "border_margin": 10,
        "nms_k": 2.6,
        "cluster_eps_mult": 3.0,   # eps = mult * median_sigma
    },
    "10x": {
        "radius_px": (6, 14),
        "peak_percentile": 99.6,
        "stripe_gate_dist": 15,
        "border_margin": 10,
        "nms_k": 2.6,
        "cluster_eps_mult": 3.0,
    },
    "4x": {
        "radius_px": (3, 9),
        "peak_percentile": 99.7,
        "stripe_gate_dist": 15,
        "border_margin": 10,
        "nms_k": 2.6,
        "cluster_eps_mult": 3.0,
    },
}
def sigmas_from_radius(radius_range, step=0.8):
    rmin, rmax = radius_range
    smin = rmin / 2.8
    smax = rmax / 2.8
    return np.arange(smin, smax + 1e-6, step)

#---------------Rotate Image-----------------
#takes in grayscale, detects lines then estimates angle of rotation
def estimate_rotation_angle(imGray, debug=False):
    edges = cv2.Canny(imGray, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=200, maxLineGap=20)
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        if angle > 90: angle -= 180
        if angle < -90: angle += 180
        if abs(angle) < 80:
            angles.append(angle)
    if not angles:
        return 0.0
    angle_med = float(np.median(angles))
    if debug:
        print("Estimated stripe angle:", angle_med)
    return angle_med

def rotate_image(img, angle, border_value=0):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )

def rotated_valid_mask(shape_hw, angle):
    h, w = shape_hw
    ones = np.ones((h, w), dtype=np.uint8) * 255
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    valid = cv2.warpAffine(
        ones, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return valid
#-----------------Mask--------------------
def stripe_mask_from_rotated(gray_rot):
    blur = cv2.GaussianBlur(gray_rot, (7, 7), 2)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # pick the mask that corresponds to stripes (keep bright or dark stripes properly)
    if np.mean(gray_rot[th == 255]) < np.mean(gray_rot[th == 0]):
        th = cv2.bitwise_not(th)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (65, 7))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    nb_components, labels, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    mask = np.zeros_like(opened)
    area_threshold = gray_rot.shape[0] * gray_rot.shape[1] * 0.0005
    for i in range(1, nb_components):
        if stats[i, cv2.CC_STAT_AREA] >= area_threshold:
            mask[labels == i] = 255
    return mask

#---------preprocess image--------------------
def suppress_stripes(gray):
    gray = gray.astype(np.float32)

    stripe_bg = cv2.GaussianBlur(gray, (1, 41), 0)

    # Prevent division explosions
    stripe_bg[stripe_bg < 1] = 1

    out = gray / stripe_bg

    out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
    return out.astype(np.uint8)

def preprocess_for_cells(img_color, stripe_mask):
    """
    Float-safe preprocessing for LoG cell detection.
    Output: uint8 image, ready for LoG
    """

    # 1. Convert to grayscale
    if img_color.ndim == 3:
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_color.copy()

    gray = gray.astype(np.float32)

    # 2. Apply stripe mask (FLOAT SAFE)
    if stripe_mask is not None:
        gray *= (stripe_mask.astype(np.float32) / 255.0)

    # 3. Stripe suppression
    gray = suppress_stripes(gray)

    # 4. Mild denoising (blob-preserving)
    gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2, sigmaY=1.2)

    # 5. Remove slow illumination background (heat-map flattening)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=20, sigmaY=20)
    gray = gray - background

    # 6. Contrast normalization (FINAL)
    lo, hi = np.percentile(gray, (1, 98.5))
    gray = np.clip(gray, lo, hi)
    gray = (gray - lo) / (hi - lo + 1e-6)
    gray = (255 * gray).astype(np.uint8)

    return gray
#---------Heat Map--------------
def detect_cells_log(pp, sigmas, peak_percentile=99.6):
    responses = []
    best_sigma = np.zeros_like(pp, dtype=np.float32)
    best_resp = np.zeros_like(pp, dtype=np.float32)

    pp32 = pp.astype(np.float32)

    for sigma in sigmas:
        log = cv2.GaussianBlur(pp32, (0, 0), float(sigma))
        log = cv2.Laplacian(log, cv2.CV_32F)
        log = np.abs(log) * (float(sigma) ** 2)

        log = reject_elongated(log)

        mask = log > best_resp
        best_resp[mask] = log[mask]
        best_sigma[mask] = float(sigma)

    # local background subtraction
    bg = cv2.GaussianBlur(best_resp, (0, 0), sigmaX=10)
    log_norm = best_resp - bg

    # peak detection on log_norm
    dilated = cv2.dilate(log_norm, np.ones((3, 3), np.uint8))
    thr = np.percentile(log_norm, peak_percentile)
    peaks = (log_norm == dilated) & (log_norm > thr)

    ys, xs = np.where(peaks)
    for x, y in zip(xs, ys):
        responses.append({
            "x": int(x),
            "y": int(y),
            "sigma": float(best_sigma[y, x]),
            "response": float(log_norm[y, x]),
        })

    return responses



    #Focus scoring

#-------Filtering-----------------
def stripe_centerlines(stripe_mask, row_thresh=10):
    rows = np.where(stripe_mask.mean(axis=1) > row_thresh)[0]
    if len(rows) == 0:
        return np.array([], dtype=int)

    centers = []
    start = rows[0]
    prev = rows[0]
    for r in rows[1:]:
        if r == prev + 1:
            prev = r
        else:
            centers.append((start + prev) // 2)
            start = prev = r
    centers.append((start + prev) // 2)
    return np.array(centers, dtype=int)

def gate_by_stripe_centers(cells, stripe_mask, max_dist=6):
    centers = stripe_centerlines(stripe_mask)
    if len(centers) == 0:
        return cells
    gated = []
    for c in cells:
        if np.min(np.abs(centers - c["y"])) <= max_dist:
            gated.append(c)
    return gated

def reject_elongated(log_resp, anisotropy_thresh=2.0):
    """
    Suppress elongated (stripe-like) responses.
    Keeps isotropic (cell-like) blobs.
    """
    log_resp = log_resp.astype(np.float32)

    gx = cv2.Sobel(log_resp, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(log_resp, cv2.CV_32F, 0, 1, ksize=3)

    mag_x = np.abs(gx)
    mag_y = np.abs(gy)

    ratio = (np.maximum(mag_x, mag_y) + 1e-6) / (np.minimum(mag_x, mag_y) + 1e-6)

    log_resp[ratio > anisotropy_thresh] = 0
    return log_resp

def gate_by_valid_distance(cells, valid_mask, margin_px=2, r_scale=2.8):
    """
    Keep detections whose *entire circle* stays inside the valid region.
    Uses distance-to-invalid-border in pixels.
    """
    if valid_mask is None:
        return cells

    v = (valid_mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(v, cv2.DIST_L2, 5)  # dist to nearest 0 pixel

    out = []
    for c in cells:
        x, y = c["x"], c["y"]
        r = r_scale * float(c["sigma"])
        if dist[y, x] >= (r + margin_px):
            out.append(c)
    return out


def classify_focus(cells, sigma_thresh):
    """
    Labels cells as in-focus or out-of-focus.
    """
    for c in cells:
        c["focus"] = "in" if c["sigma"] <= sigma_thresh else "out"
    return cells

def filter_min_distance(cells, k=1.1, r_scale=2.8):
    """
    Scale-aware NMS in PIXELS.
    rad_px = k * max(radius_px_of_two_cells)
           = k * r_scale * max(sigma)
    """
    kept = []
    for c in sorted(cells, key=lambda x: -x["response"]):
        x, y, s = c["x"], c["y"], float(c["sigma"])
        ok = True
        for kpt in kept:
            dx = x - kpt["x"]
            dy = y - kpt["y"]
            d2 = dx * dx + dy * dy

            rad = k * r_scale * max(s, float(kpt["sigma"]))  # <-- KEY FIX
            if d2 < rad * rad:
                ok = False
                break
        if ok:
            kept.append(c)
    return kept


# -------------Cluster Detection------------------------
def detect_clusters(cells, cluster_dist):
    clusters = []
    used = set()

    for i, c in enumerate(cells):
        if i in used:
            continue
        cluster = [c]
        used.add(i)

        for j, o in enumerate(cells):
            if j in used:
                continue
            if math.hypot(c["x"] - o["x"], c["y"] - o["y"]) < cluster_dist:
                cluster.append(o)
                used.add(j)

        clusters.append(cluster)

    return clusters

def find_clusters_dbscan(cells, eps, min_samples=2):
    """
    cells: list of dicts with keys ['x', 'y', 'sigma', 'focus']
    eps: clustering distance (in pixels)
    min_samples: minimum cells to form a cluster

    Returns:
        labels: array of cluster labels (-1 = isolated cell)
        n_clusters: number of clusters
    """
    if len(cells) == 0:
        return np.array([]), 0

    points = np.array([[c["x"], c["y"]] for c in cells])

    db = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean"
    ).fit(points)

    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    return labels, n_clusters

def cluster_eps_from_cells(cells, mult=1.6, r_scale=2.8, fallback=25):
    if not cells:
        return fallback
    med_sigma = float(np.median([c["sigma"] for c in cells]))
    med_r = r_scale * med_sigma
    return mult * med_r


def nms_within_clusters(cells, labels, k=2.2):
    """
    Run scale-aware NMS inside each DBSCAN cluster separately.
    Keeps clusters intact while removing duplicate peaks per cell.
    """
    # group indices by label
    groups = {}
    for idx, lbl in enumerate(labels):
        groups.setdefault(int(lbl), []).append(idx)

    kept = []
    for lbl, idxs in groups.items():
        # run NMS on this group only
        group_cells = [cells[i] for i in idxs]
        group_cells = filter_min_distance(group_cells, k=k)
        kept.extend(group_cells)

    return kept

#-----------------Circle Drawing------------------
def count_results(cells, clusters):
    in_focus = [c for c in cells if c["focus"] == "in"]
    out_focus = [c for c in cells if c["focus"] == "out"]
    cluster_cells = sum(len(cl) for cl in clusters if len(cl) > 1)

    return {
        "in_focus_cells": len(in_focus),
        "out_of_focus_cells": len(out_focus),
        "clusters": sum(1 for cl in clusters if len(cl) > 1),
        "cells_in_clusters": cluster_cells
    }

def draw_cells_and_clusters(
    base_img,
    in_focus_cells,
    out_of_focus_cells,
    draw_hulls=True
):
    """
    Draws:
      - single in-focus cells (green)
      - clustered in-focus cells (red)
      - out-of-focus cells (blue)
    """

    out = base_img.copy()

    # In-focus cells ----------
    for c in in_focus_cells:
        x, y = int(c["x"]), int(c["y"])
        r = int(2.8 * c["sigma"])

        if c["cluster"] == -1:
            color = (0, 255, 0)      # green = single cell
        else:
            color = (0, 0, 255)      # red = clustered

        cv2.circle(out, (x, y), r, color, 2)
        cv2.circle(out, (x, y), 2, color, -1)

    #  Out-of-focus cells ----------
    for c in out_of_focus_cells:
        x, y = int(c["x"]), int(c["y"])
        r = int(3.4 * c["sigma"])

        cv2.circle(out, (x, y), r, (255, 0, 0), 2)   # blue
        cv2.circle(out, (x, y), 2, (255, 0, 0), -1)

    # Cluster hulls (optional, recommended) ----------
    if draw_hulls:
        clusters = {}
        for c in in_focus_cells:
            lbl = c["cluster"]
            if lbl == -1:
                continue
            clusters.setdefault(lbl, []).append((c["x"], c["y"]))

        for pts in clusters.values():
            if len(pts) < 3:
                continue
            pts = np.array(pts, dtype=np.int32)
            hull = cv2.convexHull(pts)
            cv2.polylines(out, [hull], True, (0, 0, 180), 2)

    return out


#----Image Select------
def pick_image_file():
    root = Tk()
    root.withdraw()  # hide empty tkinter window
    file_path = filedialog.askopenfilename(
        title="Select cell image",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff"),
            ("All files", "*.*")
        ]
    )
    root.destroy()
    return file_path

#--------------Main---------------
def main(
    min_sigma=5.0,
    max_sigma=8.0,
    num_scales=10,
    focus_sigma_thresh=4.5,
    min_dist=6,
    cluster_eps=18,
    min_cluster_size=2,
    debug=True
):
    mag = "20x"
    cfg = MAG_CONFIGS[mag]
    sigmas = sigmas_from_radius(cfg["radius_px"], step=0.6)  # slightly finer

    # --------------------------------------------------
    # Load image
    # --------------------------------------------------

    image_path = pick_image_file()


    img_color = cv2.imread(image_path)
    if not image_path:
        print("No file selected. Exiting.")
        return

    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------
    # Estimate rotation + rotate
    # --------------------------------------------------
    angle = estimate_rotation_angle(gray, debug=debug)

    img_rot = rotate_image(img_color, angle, border_value=(0, 0, 0))
    gray_rot = cv2.cvtColor(img_rot, cv2.COLOR_BGR2GRAY)

    valid = rotated_valid_mask(gray.shape, angle)  # 255 where pixels are real

    # --------------------------------------------------
    # Preprocessing → response-ready image
    # --------------------------------------------------
    stripe_mask = stripe_mask_from_rotated(gray_rot)
    stripe_mask = cv2.bitwise_and(stripe_mask, valid)  # keep only valid region

    pp = preprocess_for_cells(img_rot, stripe_mask)

    # --------------------------------------------------
    # Multi-scale LoG detection (PURE detection)
    # --------------------------------------------------

    pp = 255 - pp
    sigmas = np.arange(4.5, 11.0, 0.7)
    min_sigma_keep = 4.0

    cells = detect_cells_log(pp, sigmas, peak_percentile=cfg["peak_percentile"])

    if debug:
        print(f"Raw detections (pre-valid): {len(cells)}")

    cells = [c for c in cells if valid[c["y"], c["x"]] > 0]
    #cells = gate_by_valid_interior(cells, valid, margin=25)  # try 20–40

    if debug:
        print(f"After valid-mask gate: {len(cells)}")

    cells = gate_by_stripe_centers(cells, stripe_mask, max_dist=8)
    print("After stripe gating:", len(cells))

    # --------------------------------------------------
    # Focus classification (in / out)
    # --------------------------------------------------
    sig = np.array([c["sigma"] for c in cells], dtype=float)
    focus_sigma_thresh = np.percentile(sig, 60)  # top ~60% sharpest as "in"

    print(f"sigma stats: min={sig.min():.2f} med={np.median(sig):.2f} max={sig.max():.2f}")

    cells = classify_focus(cells, sigma_thresh=focus_sigma_thresh)

    # --------------------------------------------------
    # Non-maximum suppression (merge duplicates)
    # --------------------------------------------------
    cells = filter_min_distance(cells, k=cfg["nms_k"])


    if debug:
        print(f"After min-distance filtering: {len(cells)}")

    # --------------------------------------------------
    # Separate in-focus vs out-of-focus
    # --------------------------------------------------
    # Focus classification
    cells = classify_focus(cells, sigma_thresh=focus_sigma_thresh)

    # Separate
    in_focus_cells = [c for c in cells if c["focus"] == "in"]
    out_of_focus_cells = [c for c in cells if c["focus"] == "out"]

    # DBSCAN first (on in-focus candidates)
    labels, n_clusters = find_clusters_dbscan(
        in_focus_cells,
        eps=cluster_eps,
        min_samples=min_cluster_size
    )

    # Attach cluster labels
    for c, lbl in zip(in_focus_cells, labels):
        c["cluster"] = int(lbl)

    # NOW remove duplicates without breaking clusters
    in_focus_cells = nms_within_clusters(in_focus_cells, labels, k=2.4)

    # If you also want NMS on out-of-focus (optional)
    out_of_focus_cells = filter_min_distance(out_of_focus_cells, k=2.4)

    # Recombine if you need a unified list for counting
    cells = in_focus_cells + out_of_focus_cells

    # --------------------------------------------------
    # Counting
    # --------------------------------------------------
    cluster_groups = {}
    for c in in_focus_cells:
        cluster_groups.setdefault(c["cluster"], []).append(c)

    clusters = [v for k, v in cluster_groups.items() if k != -1]

    counts = count_results(cells, clusters)

    print("----- Results -----")
    for k, v in counts.items():
        print(f"{k}: {v}")

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------
    vis = draw_cells_and_clusters(
        img_rot,
        in_focus_cells,
        out_of_focus_cells,
        draw_hulls=True
    )

    cv2.imshow("Cells & Clusters", vis)
    out_img = draw_cells_and_clusters(
        img_rot,
        in_focus_cells,
        out_of_focus_cells,
        draw_hulls=True
    )

    cv2.imwrite("cell_detection_result.png", out_img)
    print("Saved: cell_detection_result.png")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return counts, vis

# run main
main(
    min_sigma=5.0,
    max_sigma=8.0,
    num_scales=10,
    focus_sigma_thresh=4.5,
    min_dist=6,
    cluster_eps=18,
    min_cluster_size=2,
    debug=True
)
