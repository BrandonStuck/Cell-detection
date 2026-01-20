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


# ----------Circle Detection------------------
def detect_hough_circles(masked_gray, min_r, max_r, param2=22):
    gb = cv2.GaussianBlur(masked_gray, (7,7), 1.5)
    if np.count_nonzero(gb) < 10:
        return []
    circles = cv2.HoughCircles(
        gb, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=max(8, int(min_r * 0.6)),
        param1=50, param2=param2,
        minRadius=max(1, int(min_r * 0.8)), maxRadius=int(max_r * 1.2)
    )
    res = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0]:
            res.append((int(c[0]), int(c[1]), int(c[2])))
    return res

def detect_outer_contours(gray, min_r, max_r):
    """Detect circular outer contours around inner circles."""
    edges = cv2.Canny(gray, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    res = []
    for cnt in contours:
        (x, y), r = cv2.minEnclosingCircle(cnt)
        r = int(r)
        if r < min_r or r > max_r * 3:  # expanded range to catch full outer walls
            continue
        area = cv2.contourArea(cnt)
        circ = 4 * math.pi * area / (cv2.arcLength(cnt, True) ** 2 + 1e-6)
        if circ > 0.6:  # roughly circular
            res.append((int(x), int(y), r))


    return res

#------------Merge Duplicated Circles--------------------
def deduplicate_circles(circles, min_r=6, max_r=30, center_thresh=0.5, radius_thresh=0.4):
    """
    Merge duplicates and remove false detections.
    Two circles are considered the same if:
    - Their centers are closer than (center_thresh * average radius)
    - Their radii differ by less than (radius_thresh * average radius)
    """
    if not circles:
        return []

    # remove extreme radius outliers first
    circles = [(x, y, r) for (x, y, r) in circles if min_r <= r <= max_r]

    unique = []
    for (x, y, r) in circles:
        merged = False
        for i, (ux, uy, ur) in enumerate(unique):
            dist = np.hypot(x - ux, y - uy)
            if dist < center_thresh * (r + ur) / 2 and abs(r - ur) < radius_thresh * (r + ur) / 2:
                # merge (average weighted by radius)
                unique[i] = ((ux + x) / 2, (uy + y) / 2, (ur + r) / 2)
                merged = True
                break
        if not merged:
            unique.append((x, y, r))

    print(f"[DEDUP] input={len(circles)}, unique={len(unique)}")
    return unique

#-------------Eliminate False Detected Circles------------
def remove_nearby_false_circles(circles, real_circles, min_distance_factor=1.2):
    """
    Remove candidate circles that are too close to confirmed real droplets.
    Keeps isolated circles.
    """
    filtered = []
    for c in circles:
        too_close = False
        for rc in real_circles:
            d = math.hypot(c[0] - rc[0], c[1] - rc[1])
            if d < min_distance_factor * (c[2] + rc[2]):
                too_close = True
                break
        if not too_close:
            filtered.append(c)
    return filtered

def is_real_cell(gray, x, y, r):
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


def filter_by_min_distance(circles, min_distance):
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
def main(min_r=6, max_r=30):
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
    hough_circles = detect_hough_circles(masked_gray, min_r, max_r, param2=30)
    contour_circles = detect_outer_contours(gray_rot, min_r, max_r)
    combined = hough_circles + contour_circles

    # --- Filter real droplets ---
    combined = filter_by_min_distance(combined, min_distance=15)
    real_circles = [c for c in combined if is_real_droplet(gray_rot, *c)]
    print(f"Real droplets detected: {len(real_circles)}")

    # --- Remove nearby false detections ---
    filtered_circles = remove_nearby_false_circles(combined, real_circles, min_distance_factor=1.2)

    print(f"Filtered circles after removing too-close false detections: {len(filtered_circles)}")

    # --- Merge real + filtered circles
    unique_circles = deduplicate_circles(real_circles + filtered_circles, min_r=min_r, max_r=max_r,
                                         center_thresh=6, radius_thresh=4)



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
main(min_r=6, max_r=30)
