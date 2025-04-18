import cv2
import numpy as np
from pathlib import Path


def run_canny_edge_detection_gui(image_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv_image)

    window_name = "Canny Edge Detection (HSV)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Trackbars
    cv2.createTrackbar("Threshold1", window_name, 50, 500, lambda x: None)
    cv2.createTrackbar("Threshold2", window_name, 150, 500, lambda x: None)
    cv2.createTrackbar("Aperture Size (x2+3)", window_name, 1, 2, lambda x: None)  # 3, 5, 7
    cv2.createTrackbar("Use L2 Gradient", window_name, 0, 1, lambda x: None)       # False (0), True (1)
    cv2.createTrackbar("Gaussian Blur (odd)", window_name, 0, 10, lambda x: None)  # 0 = no blur, then 1,3,5...

    while True:
        t1 = cv2.getTrackbarPos("Threshold1", window_name)
        t2 = cv2.getTrackbarPos("Threshold2", window_name)
        aperture_index = cv2.getTrackbarPos("Aperture Size (x2+3)", window_name)
        aperture_size = 2 * aperture_index + 3  # 3, 5, 7
        use_l2 = cv2.getTrackbarPos("Use L2 Gradient", window_name) == 1
        blur_val = cv2.getTrackbarPos("Gaussian Blur (odd)", window_name)
        blur_kernel = blur_val * 2 + 1 if blur_val > 0 else 0  # Ensure odd and non-zero

        def process_channel(channel):
            if blur_kernel > 1:
                blurred = cv2.GaussianBlur(channel, (blur_kernel, blur_kernel), 0)
            else:
                blurred = channel
            return cv2.Canny(blurred, t1, t2, apertureSize=aperture_size, L2gradient=use_l2)

        edges_h = process_channel(h_channel)
        edges_s = process_channel(s_channel)
        edges_s = np.zeros(edges_h.shape, dtype=np.uint8)
        edges_v = process_channel(v_channel)

        # Combine
        combined_edges = cv2.bitwise_or(edges_h, edges_s)
        combined_edges = cv2.bitwise_or(combined_edges, edges_v)

        # Convert for display
        stack = lambda *imgs: np.hstack([cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in imgs])
        top_row = stack(edges_h, edges_s)
        bottom_row = stack(edges_v, combined_edges)
        all_edges = np.vstack((top_row, bottom_row))

        cv2.imshow(window_name, all_edges)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_file = Path(r"C:\Users\rkjo\OneDrive\Documents\pool_tracking\frames_output\frame_01000.png")  # Replace with your image path

    run_canny_edge_detection_gui(image_file)







