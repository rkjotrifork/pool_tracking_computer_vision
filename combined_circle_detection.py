import cv2
import numpy as np
import math
from pathlib import Path
from skimage import transform


def crop_image(image):
    x_min, y_min = 260, 300
    x_max, y_max = 1800, 800
    mask = np.zeros_like(image)
    mask[y_min:y_max, x_min:x_max] = image[y_min:y_max, x_min:x_max]
    return mask


def preprocess_and_edge(image, params):
    blur = params['blur']
    if blur > 0:
        image = cv2.GaussianBlur(image, (blur * 2 + 1, blur * 2 + 1), 0)
    return cv2.Canny(image, params['t1'], params['t2'],
                     apertureSize=(params['aperture'] * 2 + 3),
                     L2gradient=params['L2'])


def detect_circles(edges, min_r, max_r, step, threshold, max_peaks, min_dist):
    hough_radii = np.arange(min_r, max_r, step)
    if len(hough_radii) == 0:
        return []
    hough_res = transform.hough_circle(edges, hough_radii)
    accums, cx, cy, radii = transform.hough_circle_peaks(
        hough_res, hough_radii, total_num_peaks=100, threshold=threshold)

    selected = []
    for a, x, y, r in sorted(zip(accums, cx, cy, radii), reverse=True):
        if all(math.hypot(x - sx, y - sy) >= min_dist for _, sx, sy, _ in selected):
            selected.append((a, x, y, r))
            if len(selected) >= max_peaks:
                break
    return selected


def draw_circles_on_blank(edges, circles):
    output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    for _, x, y, r in circles:
        cv2.circle(output, (x, y), r, (0, 255, 0), 1)
        cv2.circle(output, (x, y), 2, (0, 0, 255), 1)
    return output


def draw_all_circles_on_rgb(base_image, all_circles):
    overlay = base_image.copy()
    for _, x, y, r in all_circles:
        cv2.circle(overlay, (x, y), r, (0, 255, 0), 2)
        cv2.circle(overlay, (x, y), 2, (0, 0, 255), 2)
    return overlay


def advanced_circle_detection_gui(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Could not load image")

    image = crop_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hue, _, _ = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2Lab))

    hue = np.zeros(gray.shape, dtype=np.uint8)
    L = np.zeros(gray.shape, dtype=np.uint8)

    channel_params = {
        "Gray": {'t1': 56, 't2': 95, 'aperture': 0, 'L2': True, 'blur': 4, 'img': gray},
        "Hue": {'t1': 23, 't2': 39, 'aperture': 0, 'L2': True, 'blur': 10, 'img': hue},
        "L": {'t1': 57, 't2': 146, 'aperture': 0, 'L2': True, 'blur': 1, 'img': L},
        "A": {'t1': 175, 't2': 260, 'aperture': 1, 'L2': False, 'blur': 6, 'img': A},
        "B": {'t1': 109, 't2': 368, 'aperture': 1, 'L2': True, 'blur': 8, 'img': B}
    }

    window_name = "Advanced Circle Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Create sliders
    cv2.createTrackbar("Min Radius", window_name, 11, 100, lambda x: None)
    cv2.createTrackbar("Max Radius", window_name, 20, 150, lambda x: None)
    cv2.createTrackbar("Radius Step", window_name, 2, 10, lambda x: None)
    cv2.createTrackbar("Threshold x100", window_name, 50, 100, lambda x: None)
    cv2.createTrackbar("Max Peaks", window_name, 16, 100, lambda x: None)
    cv2.createTrackbar("Min Dist", window_name, 10, 100, lambda x: None)

    while True:
        min_r = cv2.getTrackbarPos("Min Radius", window_name)
        max_r = cv2.getTrackbarPos("Max Radius", window_name)
        step = max(1, cv2.getTrackbarPos("Radius Step", window_name))
        threshold = cv2.getTrackbarPos("Threshold x100", window_name) / 100.0
        max_peaks = cv2.getTrackbarPos("Max Peaks", window_name)
        min_dist = cv2.getTrackbarPos("Min Dist", window_name)

        if max_r <= min_r:
            max_r = min_r + 1

        edge_images = []
        circle_overlays = []
        all_circles = []

        for name, params in channel_params.items():
            edges = preprocess_and_edge(params['img'], params)
            circles = detect_circles(edges, min_r, max_r, step, threshold, max_peaks, min_dist)
            edge_images.append(edges)
            circle_overlays.append(draw_circles_on_blank(edges, circles))
            all_circles.extend(circles)

        final_overlay = draw_all_circles_on_rgb(image, all_circles)

        # Stack into a 2x3 grid
        rows = [
            np.hstack([circle_overlays[0], circle_overlays[1], circle_overlays[2]]),
            np.hstack([circle_overlays[3], circle_overlays[4], final_overlay])
        ]
        grid = np.vstack(rows)
        cv2.imshow(window_name, grid)

        key = cv2.waitKey(50)
        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_file = Path(r"C:\Users\rkjo\OneDrive\Documents\pool_tracking\frames_output\frame_00104.png")
    advanced_circle_detection_gui(image_file)


min_radius = 11
max_radius = 20
radius_step = 2
threshold = 0.45
max_peaks = 16
min_dist = 9