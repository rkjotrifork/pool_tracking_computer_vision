from pathlib import Path
import cv2
import numpy as np
from matplotlib import pyplot as plt


def nothing(x):
    pass

def run(image_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Resize for manageability (optional)
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

    # Convert to Lightness channel
    L, _, _ = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2Lab))

    # Create a resizable window
    cv2.namedWindow('Hough Circle Detection', cv2.WINDOW_NORMAL)

    # Create trackbars
    cv2.createTrackbar('param1', 'Hough Circle Detection', 50, 300, nothing)
    cv2.createTrackbar('param2', 'Hough Circle Detection', 30, 150, nothing)
    cv2.createTrackbar('minRadius', 'Hough Circle Detection', 0, 100, nothing)
    cv2.createTrackbar('maxRadius', 'Hough Circle Detection', 0, 200, nothing)
    cv2.createTrackbar('blur ksize', 'Hough Circle Detection', 3, 20, nothing)  # must be odd
    cv2.createTrackbar('L_min', 'Hough Circle Detection', 220, 255, nothing)
    cv2.createTrackbar('L_max', 'Hough Circle Detection', 255, 255, nothing)

    while True:
        # Get trackbar values
        param1 = cv2.getTrackbarPos('param1', 'Hough Circle Detection')
        param2 = cv2.getTrackbarPos('param2', 'Hough Circle Detection')
        minRadius = cv2.getTrackbarPos('minRadius', 'Hough Circle Detection')
        maxRadius = cv2.getTrackbarPos('maxRadius', 'Hough Circle Detection')
        ksize = cv2.getTrackbarPos('blur ksize', 'Hough Circle Detection')
        L_min = cv2.getTrackbarPos('L_min', 'Hough Circle Detection')
        L_max = cv2.getTrackbarPos('L_max', 'Hough Circle Detection')

        # Ensure blur kernel is odd and >= 3
        if ksize % 2 == 0:
            ksize += 1
        if ksize < 3:
            ksize = 3

        # Apply blur
        blurred_L = cv2.GaussianBlur(L, (ksize, ksize), 0)

        # Threshold the lightness channel
        thresholded = cv2.inRange(blurred_L, L_min, L_max)

        # Copy for drawing
        output = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)

        # Hough Circle Detection
        circles = cv2.HoughCircles(thresholded, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                   param1=param1, param2=param2,
                                   minRadius=minRadius, maxRadius=maxRadius)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)

        cv2.imshow('Hough Circle Detection', output)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_file = Path(r"C:\Users\rkjo\OneDrive\Documents\pool_tracking\frames_output\frame_00104.png")
    run(image_file)

    A = [105, 121]
    B = [112, 136]


    # --- Load image ---
    img = cv2.imread(str(image_file))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    L, A, B = cv2.split(lab)

    # --- Thresholds ---
    a_mask = cv2.inRange(A, 105, 121)
    b_mask = cv2.inRange(B, 112, 136)
    combined_mask = cv2.bitwise_and(a_mask, b_mask)

    # --- Morphological closing ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)


    # --- Find contours and extract largest blob ---
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_img = img_rgb.copy()
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        # --- Create mask of the largest blob ---
        mask_largest_blob = np.zeros_like(closed_mask)
        cv2.drawContours(mask_largest_blob, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # --- Approximate with polygon (try to get 4 sides) ---
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # --- If not 4 points, fall back to convex hull and pick 4 extreme points ---
        if len(approx) < 4 or len(approx) > 6:
            hull = cv2.convexHull(largest_contour)
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            trapezoid = np.int0(box)
        else:
            trapezoid = np.int0(approx)

        # --- Draw trapezoid (expanded) ---
        trapezoid = np.array(trapezoid, dtype=np.float32)

        # Compute centroid of the shape
        center = np.mean(trapezoid, axis=0)

        # Scale points away from the center by X%
        scale_factor = 1.05
        expanded_trapezoid = (trapezoid - center) * scale_factor + center
        expanded_trapezoid = np.int0(expanded_trapezoid)

        # --- Draw trapezoid ---
        cv2.polylines(output_img, [expanded_trapezoid], isClosed=True, color=(255, 0, 0), thickness=3)

    else:
        mask_largest_blob = np.zeros_like(closed_mask)
        trapezoid = []



    # Draw the expanded trapezoid
    cv2.polylines(output_img, [expanded_trapezoid], isClosed=True, color=(255, 0, 0), thickness=3)

    # --- Create mask for expanded trapezoid ---
    trapezoid_mask = np.zeros(closed_mask.shape, dtype=np.uint8)
    cv2.fillPoly(trapezoid_mask, [expanded_trapezoid], 255)



    # --- Plot result ---
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img_rgb)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Largest Blob")
    plt.imshow(mask_largest_blob, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Bounding Trapezoid")
    plt.imshow(output_img)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.title("Expanded Trapezoid Mask")
    plt.imshow(trapezoid_mask, cmap="gray")
    plt.axis("off")
    plt.show()
