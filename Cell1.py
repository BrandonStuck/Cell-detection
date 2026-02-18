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
          "radius_px": (15, 30),
    "sigma_step": 0.6,
    "peak_percentile": 95.7,
    "stripe_gate_k": 0.6,
    "stripe_gate_min_dist": 12,
    "border_margin": 50,
    "nms_k": 2.5,
    "focus_sigma_percentile": 70,
    "cluster_eps_mult": 3.0,
    "cluster_min_samples": 3,
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
def detect_cells_log(pp, sigmas, peak_percentile=99.0):
    pp32 = pp.astype(np.float32)

    # compute response stack
    stack = []
    for sigma in sigmas:
        log = cv2.GaussianBlur(pp32, (0, 0), sigma)
        log = cv2.Laplacian(log, cv2.CV_32F)
        log = np.abs(log) * (sigma ** 2)
        log = reject_elongated(log)
        stack.append(log)

    stack = np.stack(stack, axis=0)  # [S, H, W]

    # background removal per-scale (helps)
    for i in range(stack.shape[0]):
        bg = cv2.GaussianBlur(stack[i], (0, 0), sigmaX=10)
        stack[i] = stack[i] - bg

    # scale-space max: must be max in (x,y) AND locally max across sigma
    best_idx = np.argmax(stack, axis=0)              # [H,W]
    best_val = np.max(stack, axis=0)                 # [H,W]

    # enforce local max across neighboring sigmas
    scale_ok = np.zeros_like(best_val, dtype=bool)
    for si in range(1, stack.shape[0]-1):
        m = (best_idx == si)
        scale_ok[m] = (stack[si][m] > stack[si-1][m]) & (stack[si][m] > stack[si+1][m])

    # 2D peak detection on best_val
    dil = cv2.dilate(best_val, np.ones((3, 3), np.uint8))
    thr = np.percentile(best_val, peak_percentile)
    peaks = (best_val == dil) & (best_val > thr) & scale_ok

    ys, xs = np.where(peaks)
    out = []
    for x, y in zip(xs, ys):
        si = int(best_idx[y, x])
        out.append({
            "x": int(x),
            "y": int(y),
            "sigma": float(sigmas[si]),
            "response": float(best_val[y, x])
        })
    return out




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

def gate_by_stripe_centers_scale(cells, stripe_mask, k=0.8, radius_mult=2.8, max_px=18):
    """
    Keep detections close to stripe centerlines.
    Distance threshold = min(max_px, k * (radius_mult * sigma))
    """
    centers = stripe_centerlines(stripe_mask)
    if len(centers) == 0:
        return cells

    kept = []
    for c in cells:
        y = c["y"]
        r = radius_mult * c["sigma"]
        thr = min(max_px, k * r)
        if np.min(np.abs(centers - y)) <= thr:
            kept.append(c)
    return kept


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

def gate_by_constant_border(cells, valid_mask, buffer_px=70):
    """
    Remove detections within buffer_px of the invalid/black rotation border.
    Constant buffer in pixels (not sigma-based).
    """
    if valid_mask is None or len(cells) == 0:
        return cells

    v = (valid_mask > 0).astype(np.uint8) * 255

    # shrink valid region inward by buffer_px
    k = 2 * buffer_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    interior = cv2.erode(v, kernel, iterations=1)

    kept = []
    for c in cells:
        if interior[c["y"], c["x"]] > 0:
            kept.append(c)
    return kept


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
        r = int(3.3 * c["sigma"])

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
def main(mag="20x", debug=True):
    cfg = MAG_CONFIGS[mag]
    if debug:
        print(f"Using config: {mag} -> {cfg}")

    # --- derived from cfg ---
    sigmas = sigmas_from_radius(cfg["radius_px"], step=cfg.get("sigma_step", 0.6))

    image_path = pick_image_file()
    if not image_path:
        print("No file selected. Exiting.")
        return

    img_color = cv2.imread(image_path)
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    angle = estimate_rotation_angle(gray, debug=debug)
    img_rot = rotate_image(img_color, angle, border_value=(0, 0, 0))
    gray_rot = cv2.cvtColor(img_rot, cv2.COLOR_BGR2GRAY)
    valid = rotated_valid_mask(gray.shape, angle)

    stripe_mask = stripe_mask_from_rotated(gray_rot)
    stripe_mask = cv2.bitwise_and(stripe_mask, valid)

    pp = preprocess_for_cells(img_rot, stripe_mask)
    pp = 255 - pp

    # --- detection using cfg ---
    cells = detect_cells_log(
        pp,
        sigmas,
        peak_percentile=cfg["peak_percentile"]
    )

    if debug:
        print(f"Raw detections (pre-valid): {len(cells)}")

    # --- border filtering using cfg ---
    cells = [c for c in cells if valid[c["y"], c["x"]] > 0]
    cells = gate_by_constant_border(cells, valid, buffer_px=cfg["border_margin"])

    if debug:
        print(f"After valid-mask gate: {len(cells)}")

    # --- stripe gating using cfg ---
    cells = gate_by_stripe_centers_scale(
        cells, stripe_mask,
        k=cfg["stripe_gate_k"],
        max_px=cfg["stripe_gate_min_dist"]  # reuse key, or rename in config if you want
    )

    print("After stripe gating:", len(cells))

    # --- NMS using cfg ---
    cells = filter_min_distance(cells, k=cfg["nms_k"])
    if debug:
        print(f"After min-distance filtering: {len(cells)}")

    # --- focus using cfg ---
    sig = np.array([c["sigma"] for c in cells], dtype=float)
    if len(sig) == 0:
        print("No cells left after filtering.")
        return

    focus_sigma_thresh = np.percentile(sig, cfg.get("focus_sigma_percentile", 60))
    print(f"sigma stats: min={sig.min():.2f} med={np.median(sig):.2f} max={sig.max():.2f}")

    cells = classify_focus(cells, sigma_thresh=focus_sigma_thresh)

    # split
    in_focus_cells = [c for c in cells if c["focus"] == "in"]
    out_of_focus_cells = [c for c in cells if c["focus"] == "out"]

    # --- clustering using cfg ---
    med_sigma = np.median([c["sigma"] for c in in_focus_cells]) if in_focus_cells else 4.0
    med_radius = 2.8 * med_sigma
    cluster_eps = cfg["cluster_eps_mult"] * med_radius
    min_cluster_size = cfg["cluster_min_samples"]

    labels, n_clusters = find_clusters_dbscan(
        in_focus_cells,
        eps=cluster_eps,
        min_samples=min_cluster_size
    )

    for c, lbl in zip(in_focus_cells, labels):
        c["cluster"] = int(lbl)

    # optional: NMS within clusters (uses same cfg nms_k)
    in_focus_cells = nms_within_clusters(in_focus_cells, labels, k=cfg["nms_k"])

    # rerun clustering after within-cluster NMS
    labels2, n_clusters2 = find_clusters_dbscan(
        in_focus_cells,
        eps=cluster_eps,
        min_samples=min_cluster_size
    )
    for c, lbl in zip(in_focus_cells, labels2):
        c["cluster"] = int(lbl)

    print(f"Clusters detected: {n_clusters2}")

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
main(mag="20x", debug=True)

