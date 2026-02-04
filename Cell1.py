import cv2
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from scipy.ndimage import gaussian_laplace
from scipy.ndimage import maximum_filter
from sklearn.cluster import DBSCAN

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

def rotate_image(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


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
def preprocess_for_cells(img_color, stripe_mask):
    """
    Preprocessing for LoG-based cell detection.
    Produces a smooth response-ready grayscale image.
    """

    # 1. Convert to grayscale (green still helps, but weighted gray is safer)
    if img_color.ndim == 3:
        gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_color.copy()

    gray = gray.astype(np.float32)

    # 2. Apply stripe mask early (critical)
    gray *= (stripe_mask > 0)

    # 3. Strong denoising without edge emphasis
    #    (preserves blob centers)
    gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2, sigmaY=1.2)

    # 4. Remove slow illumination background (heat-map flattening)
    #    This is the *key* new step
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=18, sigmaY=18)
    gray = gray - background

    # 5. Normalize to stable dynamic range
    gray -= gray.min()
    gray /= (gray.max() + 1e-6)
    gray *= 255.0

    return gray.astype(np.uint8)

#---------Heat Map--------------
def detect_cells_log(pp, sigmas):
    """
    Pure detection.
    Returns raw LoG peaks with scale + response.
    """
    H, W = pp.shape
    responses = []
    best_sigma = np.zeros_like(pp, dtype=np.float32)
    best_resp = np.zeros_like(pp, dtype=np.float32)

    for sigma in sigmas:
        log = cv2.GaussianBlur(pp, (0, 0), sigma)
        log = cv2.Laplacian(log, cv2.CV_32F)
        log = np.abs(log) * (sigma ** 2)

        mask = log > best_resp
        best_resp[mask] = log[mask]
        best_sigma[mask] = sigma

    # simple peak detection
    dilated = cv2.dilate(best_resp, np.ones((3, 3)))
    peaks = (best_resp == dilated) & (best_resp > 0)

    ys, xs = np.where(peaks)
    for x, y in zip(xs, ys):
        responses.append({
            "x": int(x),
            "y": int(y),
            "sigma": float(best_sigma[y, x]),
            "response": float(best_resp[y, x])
        })

    return responses


    #Focus scoring
def classify_focus(cells, sigma_thresh):
    """
    Labels cells as in-focus or out-of-focus.
    """
    for c in cells:
        c["focus"] = "in" if c["sigma"] <= sigma_thresh else "out"
    return cells

#------------Merge Duplicated Circles--------------------

def filter_min_distance(cells, min_dist):
    kept = []
    for c in sorted(cells, key=lambda x: -x["response"]):
        if all(
            math.hypot(c["x"] - k["x"], c["y"] - k["y"]) >= min_dist
            for k in kept
        ):
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
        r = int(1.4 * c["sigma"])

        if c["cluster"] == -1:
            color = (0, 255, 0)      # green = single cell
        else:
            color = (0, 0, 255)      # red = clustered

        cv2.circle(out, (x, y), r, color, 2)
        cv2.circle(out, (x, y), 2, color, -1)

    #  Out-of-focus cells ----------
    for c in out_of_focus_cells:
        x, y = int(c["x"]), int(c["y"])
        r = int(1.4 * c["sigma"])

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



#--------------Main---------------
def main(min_sigma=2.0, max_sigma=8.0):
    #  File selection ----------------
    Tk().withdraw()
    filename = filedialog.askopenfilename(
        title="Select image",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
    )
    if not filename:
        print("No file selected.")
        return

    img_color = cv2.imread(filename)
    if img_color is None:
        print("Failed to load image.")
        return

    # Rotation ----------------
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    angle = estimate_rotation_angle(gray)
    gray_rot = rotate_image(gray, -angle)
    img_rot_color = rotate_image(img_color, -angle)

    #  Stripe mask ----------------
    mask_stripes = stripe_mask_from_rotated(gray_rot)

    # Preprocessing ----------------
    response = preprocess_for_cells(img_rot_color, mask_stripes)

    # LoG peak detection ----------------
    cells = detect_cells_log(
        response,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        threshold=0.02,
        min_distance=6
    )

    print(f"Candidate cells detected: {len(cells)}")

    # Focus classification ----------------
    cells = classify_focus(
        response,
        cells,
        inner_scale=0.6,
        outer_scale=1.6
    )

    # Split by focus ----------------
    in_focus_cells = [c for c in cells if c["focus"] >= 0.25]
    out_of_focus_cells = [c for c in cells if c["focus"] < 0.25]

    print(f"In-focus cells: {len(in_focus_cells)}")
    print(f"Out-of-focus cells: {len(out_of_focus_cells)}")

    # Cluster detection (in-focus only) ----------------
    if len(in_focus_cells) > 1:
        X = np.array([[c["x"], c["y"]] for c in in_focus_cells])
        db = DBSCAN(eps=2.5 * max_sigma, min_samples=2).fit(X)
        labels = db.labels_
    else:
        labels = np.array([-1] * len(in_focus_cells))

    for c, lbl in zip(in_focus_cells, labels):
        c["cluster"] = lbl

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Clusters detected (in-focus): {n_clusters}")

    # ---------------- Draw results ----------------
    annotated = draw_cells_and_clusters(
        img_rot_color,
        in_focus_cells,
        out_of_focus_cells,
        draw_hulls=True
    )

    # Visualization ----------------
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(gray_rot, cmap="gray")
    plt.title("Rotated grayscale")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(response, cmap="inferno")
    plt.title("LoG response map")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.title("Cells & clusters")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# run main with appropriate sigma
main(min_sigma = 2.0 , max_sigma=8.0)
