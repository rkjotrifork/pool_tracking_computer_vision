import cv2
import numpy as np
import math
from pathlib import Path
from skimage import transform
from pyinstrument import Profiler
from concurrent.futures import ThreadPoolExecutor


SCALE_FACTOR = 1  # Example: 0.5 = half resolution



def crop_image2(image):
    x_min, y_min = 260, 300
    x_max, y_max = 1800, 800
    mask = np.zeros_like(image)
    mask[y_min:y_max, x_min:x_max] = image[y_min:y_max, x_min:x_max]
    return mask

def crop_image1(image):
    # Define the polygon
    polygon = np.array([
        [385, 706],
        [1708, 655],
        [1440, 339],
        [570, 364]
    ], dtype=np.int32)

    # Create a black mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Fill the polygon on the mask
    cv2.fillPoly(mask, [polygon], 255)

    # Apply the mask to the image
    cropped = cv2.bitwise_and(image, image, mask=mask)

    return cropped

def crop_image(image):
    polygon = np.array([
        [385, 706],
        [1708, 655],
        [1440, 339],
        [570, 364]
    ], dtype=np.int32)

    # polygon = np.array([
    #     [385, 706],
    #     [1508, 755],
    #     [1340, 339],
    #     [500, 364]
    # ], dtype=np.int32)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    masked = cv2.bitwise_and(image, image, mask=mask)

    # Also return the scaled version for processing
    new_size = (int(masked.shape[1] * SCALE_FACTOR), int(masked.shape[0] * SCALE_FACTOR))
    resized = cv2.resize(masked, new_size, interpolation=cv2.INTER_AREA)
    return masked, resized



def preprocess_and_edge(image, params):
    blur = params['blur']
    if blur > 0:
        image = cv2.GaussianBlur(image, (blur * 2 + 1, blur * 2 + 1), 0)
    return cv2.Canny(image, params['t1'], params['t2'],
                     apertureSize=(params['aperture'] * 2 + 3),
                     L2gradient=params['L2'])

# def preprocess_and_edge(image, params):
#     # Scale parameters
#     blur = max(1, int(params['blur'] * SCALE_FACTOR))
#     aperture = max(0, int(params['aperture'] * SCALE_FACTOR))
#
#     # Apply Gaussian blur
#     if blur > 0:
#         ksize = blur * 2 + 1  # kernel size must be odd
#         image = cv2.GaussianBlur(image, (ksize, ksize), 0)
#
#     # Apply Canny edge detection
#     return cv2.Canny(image,
#                      params['t1'],
#                      params['t2'],
#                      apertureSize=(aperture * 2 + 3),  # 3, 5, 7, etc.
#                      L2gradient=params['L2'])



def filter_close_circles(circles, min_dist=9, max_peaks=100):
    selected = []
    for a, x, y, r in sorted(circles, reverse=True):
        if all(math.hypot(x - sx, y - sy) >= min_dist for _, sx, sy, _ in selected):
            selected.append((a, x, y, r))
            if len(selected) >= max_peaks:
                break
    return selected

def filter_close_circles_largest_radius(circles, min_dist=9, max_peaks=100):
    remaining = list(circles)
    selected = []

    while remaining and len(selected) < max_peaks:
        # Take the first circle and find all that are close to it
        base = remaining.pop(0)
        group = [base]
        bx, by = base[1], base[2]

        to_remove = []
        for i, circle in enumerate(remaining):
            _, x, y, _ = circle
            if math.hypot(bx - x, by - y) < min_dist:
                group.append(circle)
                to_remove.append(i)

        # Remove grouped ones from the remaining list (reverse order for safe removal)
        for i in reversed(to_remove):
            del remaining[i]

        # Keep the one with the largest radius from this group
        largest = max(group, key=lambda c: c[3])  # c[3] is radius
        selected.append(largest)

    return selected



def detect_circles(edges, min_r, max_r, step, threshold, max_peaks, min_dist):
    hough_radii = np.arange(min_r, max_r, step)
    if len(hough_radii) == 0:
        return []

    hough_res = transform.hough_circle(edges, hough_radii)
    accums, cx, cy, radii = transform.hough_circle_peaks(
        hough_res, hough_radii, total_num_peaks=max_peaks, threshold=threshold)

    all_candidates = list(zip(accums, cx, cy, radii))

    # Reuse shared filtering logic
    return filter_close_circles(all_candidates, min_dist=min_dist, max_peaks=max_peaks)



def draw_all_circles_on_rgb(base_image, all_circles):
    overlay = base_image.copy()
    for _, x, y, r in all_circles:
        cv2.circle(overlay, (x, y), r, (0, 255, 0), 2)
        cv2.circle(overlay, (x, y), 2, (0, 0, 255), 2)
    return overlay


def process_channel1(params):
    edges = preprocess_and_edge(params['img'], params)
    return detect_circles(edges, min_r=11, max_r=20, step=2,
                          threshold=0.4, max_peaks=19, min_dist=12)

def process_channel(params):
    edges = preprocess_and_edge(params['img'], params)

    # Scale radii and min_dist down for smaller image
    min_r = int(11 * SCALE_FACTOR)
    max_r = int(20 * SCALE_FACTOR)
    min_dist = int(12 * SCALE_FACTOR)
    step = max(1, int(2 * SCALE_FACTOR))  # Ensure step is at least 1

    circles = detect_circles(edges, min_r=min_r, max_r=max_r, step=step,
                             threshold=0.4, max_peaks=19, min_dist=min_dist)

    # Rescale coordinates + radius back to original image space
    scaled_circles = [(a,
                       int(x / SCALE_FACTOR),
                       int(y / SCALE_FACTOR),
                       int(r / SCALE_FACTOR))
                      for (a, x, y, r) in circles]

    return scaled_circles



def stream_video(video_path: Path, output_dir: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not open video.")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    cv2.namedWindow("Circle Detection", cv2.WINDOW_NORMAL)

    counter = -1
    while True:
        counter += 1
        ret, frame = cap.read()
        if not ret:
            break

        if counter % 10 != 0:
            continue

        cropped_fullres, cropped_scaled = crop_image(frame)

        cropped = cropped_scaled
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        L, A, B = cv2.split(cv2.cvtColor(cropped, cv2.COLOR_BGR2Lab))

        channel_params = {
            "Gray": {'t1': 56, 't2': 95, 'aperture': 0, 'L2': True, 'blur': 4, 'img': gray},
            "L": {'t1': 57, 't2': 146, 'aperture': 0, 'L2': True, 'blur': 1, 'img': L},
            "A": {'t1': 175, 't2': 260, 'aperture': 1, 'L2': False, 'blur': 6, 'img': A},
            "B": {'t1': 109, 't2': 368, 'aperture': 1, 'L2': True, 'blur': 8, 'img': B}
        }

        # channel_params = {
        #     "Gray": {'t1': 56, 't2': 95, 'aperture': 0, 'L2': True, 'blur': 1, 'img': gray},
        #     "L": {'t1': 57, 't2': 146, 'aperture': 0, 'L2': True, 'blur': 1, 'img': L},
        #     "A": {'t1': 175, 't2': 260, 'aperture': 1, 'L2': False, 'blur': 1, 'img': A},
        #     "B": {'t1': 109, 't2': 368, 'aperture': 1, 'L2': True, 'blur': 1, 'img': B}
        # }

        # --- Parallel Circle Detection ---
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(process_channel, channel_params.values())

        all_circles = []
        for circles in results:
            all_circles.extend(circles)

        # Filter combined results to remove duplicates/close ones
        # all_circles = filter_close_circles(all_circles, min_dist=12, max_peaks=16)
        all_circles = filter_close_circles_largest_radius(all_circles, min_dist=12, max_peaks=16)

        # result_frame = draw_all_circles_on_rgb(cropped, all_circles)
        result_frame = draw_all_circles_on_rgb(cropped_fullres, all_circles)

        cv2.imshow("Circle Detection", result_frame)

        #Save minimum bounding boxes with alpha channel turned off outside circle for each circle
        # and the filenames should be frame number and x y coordinates of circle and radius

        # Save each detected circle with transparency outside
        save_circles = False
        if save_circles:
            for (_, x, y, r) in all_circles:
                x, y, r = int(x), int(y), int(r)
                size = r * 2
                top_left_x = max(x - r, 0)
                top_left_y = max(y - r, 0)
                bottom_right_x = min(x + r, cropped_fullres.shape[1])
                bottom_right_y = min(y + r, cropped_fullres.shape[0])

                roi = cropped_fullres[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

                # Create alpha mask
                alpha = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
                center = (roi.shape[1] // 2, roi.shape[0] // 2)
                cv2.circle(alpha, center, r, (255), thickness=-1)

                # Add alpha channel to image
                bgr = roi.copy()
                bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
                bgra[:, :, 3] = alpha

                # Save to file
                filename = f"frame{counter}_x{x}_y{y}_r{r}.png"
                cv2.imwrite(str(output_dir / filename), bgra)


        if cv2.waitKey(1) == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = Path(r"C:\Users\rkjo\OneDrive\Documents\pool_tracking\VID20250310120030.mp4")
    # video_path = Path(r"C:\Users\rkjo\OneDrive\Documents\pool_tracking\short_4_1_dropped.mp4")
    # video_path = Path(r"C:\Users\rkjo\OneDrive\Documents\pool_tracking\VID20250319123153.mp4")


    circles_output_dir = Path(r"C:\Users\rkjo\OneDrive\Documents\pool_tracking\circles_output")

    profiler = Profiler()
    profiler.start()

    stream_video(video_path, circles_output_dir)

    profiler.stop()
    profiler.print()

    # 385, 706
    # 1708, 655
    # 1440, 339
    # 570, 374
