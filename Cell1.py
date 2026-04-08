import cv2
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from scipy.ndimage import gaussian_laplace
from scipy.ndimage import maximum_filter
from sklearn.cluster import DBSCAN
import time


MAG_CONFIGS = {
    "20x_2mil": {
          "radius_px": (11, 34),
    "sigma_step": 0.6,
    "peak_percentile": 96.3,
    "stripe_gate_k": 1.0,
    "stripe_gate_min_dist": 22,
    "border_margin": 60,
    "nms_k": 1.75,
    "focus_sigma_percentile": 70,
    "cluster_eps_mult": 3.5,
    "cluster_min_samples": 2,
    },
    "20x_1mil": {
        "radius_px": (11, 34),
        "sigma_step": 0.6,
        "peak_k_mad": 3.5,
        "edge_k_mad": 3.2,
        "score_thresh": 6.0,
        "stripe_gate_k": 1.0,
        "stripe_gate_min_dist": 22,
        "border_margin": 60,
        "nms_k": 1.65,
        "focus_sigma_abs": 6.0,
        "cluster_eps_mult": 3.5,
        "cluster_min_samples": 2,
    },
    "20x_0.5mil": {
        "radius_px": (10, 28),
        "sigma_step": 0.5,
        "peak_percentile": 96.5,
        "stripe_gate_k": 1.15,
        "stripe_gate_min_dist": 26,
        "border_margin": 60,
        "nms_k": 1.85,
        "focus_sigma_percentile": 70,
        "cluster_eps_mult": 3.5,
        "cluster_min_samples": 2,
        "score_thresh": 9.5,
"min_circular_occupancy": 0.42,
"min_isotropy": 0.55,
"max_centerline_penalty": 3.2,
    },
}
def extract_patch(img, x, y, r):
    """
    Extract square patch centered at (x,y), radius r.
    Returns patch and top-left corner (x0,y0). Returns None if too close to border.
    """
    h, w = img.shape[:2]
    x0, x1 = x - r, x + r + 1
    y0, y1 = y - r, y + r + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None, None, None
    return img[y0:y1, x0:x1], x0, y0


def circular_mask(h, w, cx, cy, r_in, r_out=None):
    """
    Disk mask if r_out is None, ring mask otherwise.
    """
    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    if r_out is None:
        return d2 <= (r_in ** 2)
    return (d2 >= (r_in ** 2)) & (d2 <= (r_out ** 2))


def sample_circle_points(cx, cy, r, n=32):
    pts = []
    for t in np.linspace(0, 2*np.pi, n, endpoint=False):
        x = int(round(cx + r * np.cos(t)))
        y = int(round(cy + r * np.sin(t)))
        pts.append((x, y))
    return pts


def compute_candidate_features(gray, x, y, sigma):
    """
    Compute local patch features around one candidate.
    Uses grayscale image after preprocessing/inversion (pp), not color image.
    """
    r = max(7, int(round(2.8 * sigma)))
    patch, x0, y0 = extract_patch(gray, x, y, r)
    if patch is None:
        return None

    ph, pw = patch.shape[:2]
    cx = x - x0
    cy = y - y0

    # gradient
    gx = cv2.Sobel(patch.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx * gx + gy * gy)

    # ---- feature 1: ring contrast ----
    center_mask = circular_mask(ph, pw, cx, cy, max(2, int(0.45 * r)))
    ring_mask = circular_mask(ph, pw, cx, cy, int(0.75 * r), int(1.15 * r))

    center_mean = float(np.mean(patch[center_mask])) if np.any(center_mask) else 0.0
    ring_mean = float(np.mean(patch[ring_mask])) if np.any(ring_mask) else 0.0

    # for your inverted pp, true cells tend to have stronger ring than center
    ring_contrast = ring_mean - center_mean

    # ---- feature 2: circular edge occupancy ----
    circle_pts = sample_circle_points(cx, cy, int(round(r)), n=40)
    vals = []
    for px, py in circle_pts:
        if 0 <= px < pw and 0 <= py < ph:
            vals.append(float(gmag[py, px]))
    if len(vals) == 0:
        return None

    vals = np.array(vals, dtype=np.float32)
    gthr = np.percentile(gmag, 75) if np.any(gmag > 0) else 0.0
    circular_occupancy = float(np.mean(vals > gthr))

    # ---- feature 3: isotropy ----
    gx_abs = float(np.mean(np.abs(gx[ring_mask]))) if np.any(ring_mask) else 0.0
    gy_abs = float(np.mean(np.abs(gy[ring_mask]))) if np.any(ring_mask) else 0.0
    isotropy = min(gx_abs, gy_abs) / (max(gx_abs, gy_abs) + 1e-6)

    # ---- feature 4: centerline penalty ----
    # if gradient is overwhelmingly vertical/horizontal at center, it's often stripe junk
    local_gx = float(np.mean(np.abs(gx[max(0, cy-1):min(ph, cy+2), max(0, cx-1):min(pw, cx+2)])))
    local_gy = float(np.mean(np.abs(gy[max(0, cy-1):min(ph, cy+2), max(0, cx-1):min(pw, cx+2)])))
    centerline_penalty = max(local_gx, local_gy) / (min(local_gx, local_gy) + 1e-6)

    return {
        "ring_contrast": ring_contrast,
        "circular_occupancy": circular_occupancy,
        "isotropy": isotropy,
        "centerline_penalty": centerline_penalty,
        "r": r
    }


def score_candidate(feat):
    """
    Combine features into one score.
    Higher = more cell-like.
    """
    if feat is None:
        return -1e9

    score = 0.0
    score += 0.08 * feat["ring_contrast"]
    score += 2.2 * feat["circular_occupancy"]
    score += 1.2 * feat["isotropy"]
    score -= 0.18 * max(0.0, feat["centerline_penalty"] - 1.5)
    return score


def filter_candidates_by_score(cells, gray_for_scoring, score_thresh=4.0, debug=False):
    kept = []
    scores = []

    for c in cells:
        feat = compute_candidate_features(gray_for_scoring, c["x"], c["y"], c["sigma"])
        s = score_candidate(feat)
        c["score"] = float(s)
        if feat is not None:
            c["ring_contrast"] = feat["ring_contrast"]
            c["circular_occupancy"] = feat["circular_occupancy"]
            c["isotropy"] = feat["isotropy"]
            c["centerline_penalty"] = feat["centerline_penalty"]
        if feat is None:
            continue



    if debug and len(scores) > 0:
        arr = np.array(scores, dtype=float)
        print(f"Candidate score stats: min={arr.min():.2f} med={np.median(arr):.2f} max={arr.max():.2f}")
        print(f"After candidate scoring: {len(kept)} / {len(cells)} kept")

    valid_scores = [s for s in scores if s > -1e8]
    if debug and len(valid_scores) > 0:
        arr = np.array(valid_scores, dtype=float)
        print("Score percentiles:",
              np.percentile(arr, [1, 5, 10, 25, 50, 75, 90, 95, 99]))
    return kept

def detect_cells_from_edges(pp, stripe_safe, existing_cells=None,
                            min_area=20, max_area=120, min_dist_to_existing=12):
    gx = cv2.Sobel(pp.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(pp.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx * gx + gy * gy)

    vals = gmag[stripe_safe > 0]
    if vals.size == 0:
        return []

    thr = np.percentile(vals, 92)   # much stricter than before
    bw = (gmag > thr).astype(np.uint8) * 255
    bw = cv2.bitwise_and(bw, stripe_safe)

    # remove tiny specks
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, ker, iterations=1)

    num, labels, stats, cent = cv2.connectedComponentsWithStats(bw, 8)

    out = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue

        x, y = cent[i]
        x = int(round(x))
        y = int(round(y))

        if existing_cells is not None:
            too_close = False
            for c in existing_cells:
                dx = x - c["x"]
                dy = y - c["y"]
                if dx*dx + dy*dy < min_dist_to_existing**2:
                    too_close = True
                    break
            if too_close:
                continue

        out.append({
            "x": x,
            "y": y,
            "sigma": 4.8,   # use realistic small-cell sigma for 20x
            "response": float(area)
        })

    return out
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
        stack[i] = np.maximum(stack[i], 0)

    # scale-space max: must be max in (x,y) AND locally max across sigma
    best_idx = np.argmax(stack, axis=0)              # [H,W]
    best_val = np.max(stack, axis=0)                 # [H,W]

    # enforce local max across neighboring sigmas
    scale_ok = np.zeros_like(best_val, dtype=bool)
    for si in range(1, stack.shape[0]-1):
        m = (best_idx == si)
        eps = 1e-6
        scale_ok[m] = (stack[si][m] >= stack[si - 1][m] - eps) & (stack[si][m] >= stack[si + 1][m] - eps)

    # 2D peak detection on best_val
    dil = cv2.dilate(best_val, np.ones((3, 3), np.uint8))
    thr = np.percentile(best_val, peak_percentile)
    eps2 = 3e-4 * float(best_val.max() + 1e-6)
    peaks = (best_val >= dil - eps2) & (best_val > thr) & scale_ok

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


def reject_elongated(log_resp, anisotropy_thresh=1.5):
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
    interior = cv2.erode(v, kernel, iterations=2)

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

            rad = k * r_scale * math.sqrt(s * float(kpt["sigma"]))
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

    points = np.array([[c["x"]] for c in cells], dtype=np.float32)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
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
        r = int(2.8 * c["sigma"])

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

    t0 = time.perf_counter()
    img_color = cv2.imread(image_path)
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    angle = estimate_rotation_angle(gray, debug=debug)
    img_rot = rotate_image(img_color, angle, border_value=(0, 0, 0))
    gray_rot = cv2.cvtColor(img_rot, cv2.COLOR_BGR2GRAY)
    valid = rotated_valid_mask(gray.shape, angle)

    stripe_mask = stripe_mask_from_rotated(gray_rot)
    stripe_mask = cv2.bitwise_and(stripe_mask, valid)

    stripe_blur = cv2.GaussianBlur(stripe_mask, (0, 0), 5)
    stripe_bin = (stripe_blur > 35).astype(np.uint8) * 255

    edge_pad = 9
    k = 2 * edge_pad + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    stripe_safe = cv2.erode(stripe_bin, ker, iterations=1)

    pp = preprocess_for_cells(img_rot, stripe_blur)
    pp = 255 - pp

    cells = detect_cells_log(pp, sigmas, peak_percentile=cfg["peak_percentile"])

    print(f"Raw detections (pre-valid): {len(cells)}")

    cells = [c for c in cells if valid[c["y"], c["x"]] > 0]
    cells = gate_by_constant_border(cells, valid, buffer_px=cfg["border_margin"])
    cells = [c for c in cells if stripe_safe[c["y"], c["x"]] > 0]

    print("stripe_bin coverage:", np.mean(stripe_bin > 0))
    print("stripe_safe coverage:", np.mean(stripe_safe > 0))
    print(f"After valid-mask gate: {len(cells)}")

    cells = gate_by_stripe_centers_scale(
        cells, stripe_bin,
        k=cfg["stripe_gate_k"],
        max_px=cfg["stripe_gate_min_dist"]
    )


    print("After stripe gating:", len(cells))

    edge_cells = detect_cells_from_edges(pp, stripe_safe, existing_cells=cells)
    cells = cells + edge_cells
    # --- candidate scoring using local patch features ---
    cells = filter_candidates_by_score(
        cells,
        gray_for_scoring=pp,
        score_thresh=cfg.get("score_thresh", 7.0),
        debug=debug
    )

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

    centers = stripe_centerlines(stripe_bin)  # use binary, not blurred
    if len(centers) > 0:
        # assign each cell to nearest stripe centerline
        for c in in_focus_cells:
            c["row"] = int(np.argmin(np.abs(centers - c["y"])))
    else:
        for c in in_focus_cells:
            c["row"] = -1

    print("stripe centerlines found:", len(centers))
    print("centerlines:", centers[:20])

    row_ids = [c["row"] for c in in_focus_cells]
    print("unique row ids:", sorted(set(row_ids))[:20], "count =", len(set(row_ids)))

    sigs = [c["sigma"] for c in cells]
    print("unique sigmas sample:", sorted(set(round(s, 2) for s in sigs))[:10])

    # --- clustering using cfg ---
    med_sigma = np.median([c["sigma"] for c in in_focus_cells]) if in_focus_cells else 4.0
    med_radius = 2.8 * med_sigma
    cluster_eps = cfg["cluster_eps_mult"] * med_radius
    #cluster_eps = min(cluster_eps, 1.6 * cfg["radius_px"][1])
    min_cluster_size = cfg["cluster_min_samples"]

    next_cluster_id = 0
    for c in in_focus_cells:
        c["cluster"] = -1

    for row_id in sorted(set(c["row"] for c in in_focus_cells)):
        row_cells = [c for c in in_focus_cells if c["row"] == row_id]
        if len(row_cells) < cfg["cluster_min_samples"]:
            continue

        # IMPORTANT: within-row eps can be bigger now
        eps_row = cluster_eps * 1.6  # NEW knob: 1.0–1.8 typically

        labels, _ = find_clusters_dbscan(row_cells, eps=eps_row, min_samples=cfg["cluster_min_samples"])

        # remap row-local labels to global unique cluster IDs
        for c, lbl in zip(row_cells, labels):
            if int(lbl) == -1:
                c["cluster"] = -1
            else:
                c["cluster"] = next_cluster_id + int(lbl)

        # advance cluster id counter
        n_row_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        next_cluster_id += n_row_clusters

    total_clusters = len(set(c["cluster"] for c in in_focus_cells) - {-1})
    print(f"Clusters detected: {total_clusters}")

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
    # --------------------------------------------------
    # Visualization + Save (formatted like the example)
    # --------------------------------------------------
    vis = draw_cells_and_clusters(
        img_rot,
        in_focus_cells,
        out_of_focus_cells,
        draw_hulls=True
    )

    # --- Build a report-style canvas (white margins + title + footer) ---
    h, w = vis.shape[:2]

    top_pad = 90
    bottom_pad = 110
    side_pad = 60

    canvas_h = h + top_pad + bottom_pad
    canvas_w = w + 2 * side_pad

    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)  # white background
    canvas[top_pad:top_pad + h, side_pad:side_pad + w] = vis

    # Title (centered)
    title = mag  # e.g. "20x"
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 1.4
    title_thick = 3
    (tw, th), _ = cv2.getTextSize(title, font, title_scale, title_thick)
    tx = (canvas_w - tw) // 2
    ty = (top_pad // 2) + (th // 2) + 10
    cv2.putText(canvas, title, (tx, ty), font, title_scale, (0, 0, 0), title_thick, cv2.LINE_AA)

    # Footer text
    counts = count_results(cells, clusters)  # you already computed this above; reuse if you want
    footer = (
        f"Data: total = {counts['in_focus_cells'] + counts['out_of_focus_cells']}   in focus={counts['in_focus_cells']}  out focus={counts['out_of_focus_cells']} out/in = {round(counts['out_of_focus_cells']/counts['in_focus_cells'],2)} "
        f"clusters={counts['clusters']}  cells_in_clusters={counts['cells_in_clusters']}"
    )
    footer_scale = 0.9
    footer_thick = 2
    fx = 40
    fy = top_pad + h + 70
    cv2.putText(canvas, footer, (fx, fy), font, footer_scale, (0, 0, 0), footer_thick, cv2.LINE_AA)

    # --- Save to disk ---
    out_path = "cell_detection_report.png"
    cv2.imwrite(out_path, canvas)
    print(f"Saved: {out_path}")

    elapsed = time.perf_counter() - t0
    print(f"Total processing time: {elapsed:.3f} s")
    
    # --- Show in a window and BLOCK until user closes it ---
    win = "Cell Detection Report"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1080,720)
    cv2.imshow(win, canvas)

    # This blocks until user closes window (or presses a key if they use the window controls)
    # We'll enforce close-to-continue by polling window visibility:
    while True:
        # returns < 0 when the window is closed
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        cv2.waitKey(50)

    cv2.destroyWindow(win)
    cv2.destroyAllWindows()

    return counts, vis


# run main
main(mag="20x_0.5mil", debug=True)

