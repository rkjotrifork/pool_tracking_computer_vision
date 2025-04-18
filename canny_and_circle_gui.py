import cv2
import numpy as np
from pathlib import Path


from skimage import feature, transform, draw
import numpy as np
import matplotlib.pyplot as plt


def run_canny_edge_detection_gui(image_path: Path) -> None:
    input_image = cv2.imread(str(image_path))
    if input_image is None:
        raise ValueError(f"Could not load image from {image_path}")

    x_min, y_min = 260, 300
    x_max, y_max = 1800, 800

    # make all outside of the rectangle black
    mask = np.zeros(input_image.shape, dtype=np.uint8)
    mask[y_min:y_max, x_min:x_max] = input_image[y_min:y_max, x_min:x_max]
    input_image = mask


    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    h, s, v = cv2.split(cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV))

    L, A, B = cv2.split(cv2.cvtColor(input_image, cv2.COLOR_BGR2Lab))

    image = A

    window_name = "Canny Edge Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Allow resizing to see full image

    # Create trackbars
    cv2.createTrackbar("Threshold1", window_name, 50, 500, lambda x: None)
    cv2.createTrackbar("Threshold2", window_name, 150, 500, lambda x: None)
    cv2.createTrackbar("Aperture Size (x2+3)", window_name, 1, 2, lambda x: None)  # 0,1,2 -> 3,5,7
    cv2.createTrackbar("Use L2 Gradient", window_name, 0, 1, lambda x: None)       # 0 = False, 1 = True
    cv2.createTrackbar("Gaussian Blur (odd)", window_name, 0, 10, lambda x: None)  # 0 = no blur

    while True:
        # Read trackbar values
        t1 = cv2.getTrackbarPos("Threshold1", window_name)
        t2 = cv2.getTrackbarPos("Threshold2", window_name)
        aperture_index = cv2.getTrackbarPos("Aperture Size (x2+3)", window_name)
        aperture_size = 2 * aperture_index + 3  # 3, 5, 7
        use_l2 = cv2.getTrackbarPos("Use L2 Gradient", window_name) == 1
        blur_val = cv2.getTrackbarPos("Gaussian Blur (odd)", window_name)
        blur_kernel = blur_val * 2 + 1 if blur_val > 0 else 0

        # Apply optional Gaussian blur
        if blur_kernel > 1:
            blurred_image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
        else:
            blurred_image = image

        # Canny edge detection
        edges = cv2.Canny(blurred_image, t1, t2, apertureSize=aperture_size, L2gradient=use_l2)

        # Show result
        cv2.imshow(window_name, edges)

        # Exit on ESC
        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()

    return edges

import math

def run_hough_circle_gui(edges: np.ndarray) -> None:
    window_name = "Hough Circle Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Initial slider values
    cv2.createTrackbar("Min Radius", window_name, 10, 200, lambda x: None)
    cv2.createTrackbar("Max Radius", window_name, 50, 300, lambda x: None)
    cv2.createTrackbar("Radius Step", window_name, 2, 10, lambda x: None)
    cv2.createTrackbar("Threshold (x100)", window_name, 70, 100, lambda x: None)
    cv2.createTrackbar("Max Peaks", window_name, 16, 100, lambda x: None)
    cv2.createTrackbar("Min Dist", window_name, 10, 100, lambda x: None)  # New slider

    while True:
        min_r = cv2.getTrackbarPos("Min Radius", window_name)
        max_r = cv2.getTrackbarPos("Max Radius", window_name)
        step = max(1, cv2.getTrackbarPos("Radius Step", window_name))
        threshold = cv2.getTrackbarPos("Threshold (x100)", window_name) / 100.0
        max_peaks = max(1, cv2.getTrackbarPos("Max Peaks", window_name))
        min_dist = cv2.getTrackbarPos("Min Dist", window_name)

        if max_r <= min_r:
            max_r = min_r + 1

        hough_radii = np.arange(min_r, max_r, step)
        if len(hough_radii) == 0:
            continue

        hough_res = transform.hough_circle(edges, hough_radii)

        # Get MANY peaks first
        accums, cx, cy, radii = transform.hough_circle_peaks(
            hough_res, hough_radii,
            total_num_peaks=100,  # intentionally large
            threshold=threshold
        )

        # Custom non-max suppression: filter by min distance
        selected = []
        for a, x, y, r in sorted(zip(accums, cx, cy, radii), reverse=True):
            too_close = False
            for _, sx, sy, _ in selected:
                dist = math.hypot(x - sx, y - sy)
                if dist < min_dist:
                    too_close = True
                    break
            if not too_close:
                selected.append((a, x, y, r))
            if len(selected) >= max_peaks:
                break

        # Draw the filtered circles
        output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        for _, x, y, r in selected:
            cv2.circle(output, (x, y), r, (0, 255, 0), 1)
            cv2.circle(output, (x, y), 2, (0, 0, 255), 1)

        cv2.imshow(window_name, output)

        key = cv2.waitKey(30)
        if key == 27:
            break

    cv2.destroyWindow(window_name)




if __name__ == "__main__":
    image_file = Path(r"C:\Users\rkjo\OneDrive\Documents\pool_tracking\frames_output\frame_00104.png")  # Replace with your image path
    edges = run_canny_edge_detection_gui(image_file)


    run_hough_circle_gui(edges)

    # gray
    # t1 56, t2 95, aperture 0, L2 true, blur 4

    # hue
    # t1 23, t2 39, aperture 0, L2 true, blur 10

    # L
    # t1 57, t2 146, aperture 0, L2 true, blur 1

    # A
    # t1 175, t2 260, aperture 1, L2 False, blur 6

    # B
    # t1 109, t2 368, aperture 1, L2 True, blur 8