import cv2
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

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
def preprocess_for_cells(img, stripe_mask):
    """
    Extracts the green channel, boosts rim contrast using CLAHE,
    then applies Difference-of-Gaussians before masking.
    """

    # 1. Use green channel (cell rims are most visible here)
    gray = img[:, :, 1]

    # 2. Local contrast boost (pre-mask to avoid local context loss)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # 3. Apply stripe mask
    masked = cv2.bitwise_and(gray, gray, mask=stripe_mask)

    # 4. Difference of Gaussians (enhances soft circular rims)
    blur_small = cv2.GaussianBlur(masked, (3,3), 0.5)
    blur_large = cv2.GaussianBlur(masked, (9,9), 3)
    dog = cv2.subtract(blur_small, blur_large)

    return dog

# ----------Circle Detection------------------
def detect_hough_circles(cell_preprocessed, min_r, max_r):
    # If the region has no content, skip
    if np.count_nonzero(cell_preprocessed) < 50:
        return []

    circles = cv2.HoughCircles(
        cell_preprocessed,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(10, int(min_r * 1.2)),
        param1=80,      # low Canny threshold
        param2=20,       # VERY important: sensitive threshold for weak rims
        minRadius=min_r,
        maxRadius=max_r
    )

    if circles is None:
        return []

    circles = np.int32(np.round(circles[0]))
    return [(x, y, r) for (x, y, r) in circles]

def detect_outer_contours(gray, min_r, max_r):
    edges = cv2.Canny(gray, 25, 60)
    edges = cv2.dilate(edges, None, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    res = []
    for cnt in contours:
        (x, y), r = cv2.minEnclosingCircle(cnt)
        r = int(r)

        if r < min_r or r > max_r * 3:
            continue

        area = cv2.contourArea(cnt)
        if area < 8:
            continue

        # circularity check
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            continue

        circ = 4 * math.pi * area / (peri * peri)
        if circ > 0.55:
            res.append((int(x), int(y), r))

    return res


#------------Merge Duplicated Circles--------------------

def dedupe_color(gray, x, y, r):
    """
    Returns True if the circle at (x, y) with radius r appears to be a real droplet:
    - Bright center
    - Dark outer ring
    """
    num_samples = 20
    center_values = []
    outer_values = []

    # sample center (half radius)
    for t in np.linspace(0, 2*np.pi, num_samples):
        px = int(x + 0.5*r*np.cos(t))
        py = int(y + 0.5*r*np.sin(t))
        if 0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]:
            center_values.append(gray[py, px])
    if not center_values:
        return False
    center_brightness = np.mean(center_values)

    # sample outer ring (slightly outside radius)
    for t in np.linspace(0, 2*np.pi, num_samples):
        px = int(x + 1.1*r*np.cos(t))
        py = int(y + 1.1*r*np.sin(t))
        if 0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]:
            outer_values.append(gray[py, px])
    if not outer_values:
        return False
    ring_darkness = np.mean(outer_values)

    # Real droplets: bright center + darker ring
    return (center_brightness - ring_darkness) > 18  # adjust threshold as needed


def dedupe_min_distance(circles, min_distance):
    """
    Removes circles that are too close to any other circle.
    Only keeps circles that are at least `min_distance` apart.

    circles: list of (x, y, r)
    min_distance: minimum allowed distance between circle centers
    """
    filtered = []
    for c in circles:
        x, y, r = c
        too_close = False
        for fc in filtered:
            fx, fy, fr = fc
            d = math.hypot(x - fx, y - fy)
            if d < min_distance:
                too_close = True
                break
        if not too_close:
            filtered.append(c)
    return filtered

def merge(circles, gray, min_distance):
    """
    Combines color-based validation and distance-based deduplication.

    circles: list of (x, y, r)
    gray: grayscale image
    min_distance: minimum distance between circle centers
    """

    # Step 1: color / contrast filtering
    color_filtered = []
    for (x, y, r) in circles:
        if dedupe_color(gray, x, y, r):
            color_filtered.append((x, y, r))

    # Step 2: spatial deduplication
    final_circles = dedupe_min_distance(color_filtered, min_distance)

    return final_circles


# -------------Cluster Detection------------------------
def circles_overlap(c1, c2, overlap_factor=1.65):

    (x1, y1, r1) = c1
    (x2, y2, r2) = c2
    d = math.hypot(x1 - x2, y1 - y2)
    # using edge-to-edge: overlap if center distance <= (r1 + r2) * overlap_factor

    return d <= (r1 + r2) * overlap_factor


def find_clusters_dbscan(circles, overlap_factor=1.65):
    visited = set()
    clusters = []

    for i in range(len(circles)):
        if i in visited:
            continue

        stack = [i]
        cluster = []

        while stack:
            idx = stack.pop()
            if idx in visited:
                continue
            visited.add(idx)
            cluster.append(idx)

            # expand cluster by overlap
            for j in range(len(circles)):
                if j not in visited and circles_overlap(circles[idx], circles[j], overlap_factor):
                    stack.append(j)

        if len(cluster) > 1:
            clusters.append(cluster)

    return clusters


#-----------------Circle Drawing------------------
def draw_clusters_and_circles(base_img, unique_circles, clusters):
    out = base_img.copy()
    # draw all unique circles in green
    for (x, y, r) in unique_circles:
        cv2.circle(out, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.circle(out, (int(x), int(y)), 2, (0, 255, 0), -1)

    # draw cluster members in red and connect them
    for cid, cluster in enumerate(clusters, start=1):
        color = (0, 0, 255)  # red
        # draw lines between all pairs in cluster
        for i, j in itertools.combinations(cluster, 2):
            x1, y1, _ = unique_circles[i]
            x2, y2, _ = unique_circles[j]
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, cv2.LINE_AA)
        # draw the circles again as red so they stand out
        for i in cluster:
            x, y, r = unique_circles[i]
            cv2.circle(out, (int(x), int(y)), int(r), color, 2)
            cv2.circle(out, (int(x), int(y)), 3, (0, 255, 255), -1)  # center marker

    return out


#--------------Main---------------
def main(min_r, max_r):
    # select file
    Tk().withdraw()
    filename = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
    if not filename:
        print("No file selected.")
        return

    img_color = cv2.imread(filename)
    if img_color is None:
        print("Failed to open:", filename)
        return

    # estimate rotation and rotate
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    angle = estimate_rotation_angle(gray)
    gray_rot = rotate_image(gray, -angle)
    img_rot_color = rotate_image(img_color, -angle)

    # build stripe mask (used to focus detection to stripe areas)
    mask_stripes = stripe_mask_from_rotated(gray_rot)
    masked_gray = cv2.bitwise_and(gray_rot, gray_rot, mask=mask_stripes)

    # --- detect circles ---
    pp = preprocess_for_cells(img_color, mask_stripes)
    hough_circles = detect_hough_circles(pp, min_r, max_r)
    contour_circles = detect_outer_contours(pp, min_r, max_r)
    combined = hough_circles + contour_circles

    # --- Filter real droplets --
    real_circles = [c for c in combined if dedupe_color(gray_rot, *c)]

        # --- Step 1: color-based filtering ---
    real_circles = [c for c in combined if dedupe_color(gray_rot, *c)]
    print(f"Cells after color filter: {len(real_circles)}")

        # --- Step 2: distance-based dedupe ---
    if real_circles:
        avg_r = np.mean([r for (_, _, r) in real_circles])
        min_distance = 2.0 * avg_r
    else:
        min_distance = 0

    unique_circles = dedupe_min_distance(real_circles, min_distance)

    print(f"Unique cells: {len(unique_circles)}")

    # Print detection counts
    print(f"Hough detections: {len(hough_circles)}")
    print(f"Contour detections: {len(contour_circles)}")
    print(f"Unique circles (after dedupe): {len(unique_circles)}")

    # --- cluster detection
    clusters = find_clusters_dbscan(unique_circles)
    cluster_sizes = [len(c) for c in clusters]
    print(f"Number of clusters (size>1): {len(clusters)}")
    print("Cluster sizes:", cluster_sizes)

    # draw annotated result
    annotated = draw_clusters_and_circles(img_rot_color, unique_circles, clusters)

    #Plot with 3 diagrams
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(gray_rot, cmap='gray')
    plt.title("Rotated grayscale")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask_stripes, cmap='gray')
    plt.title("Stripe mask")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    # convert BGR to RGB for matplotlib
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.title("Detected Clusters & Connections")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


# run main with provided min/max radii
main(min_r=12, max_r=30)
