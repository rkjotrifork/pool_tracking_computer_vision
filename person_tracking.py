import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import time

#TODO add threshold for certainty of when something is detected as a person

# Flags to toggle features
USE_TRACKING = True  # Toggle ID tracking
USE_MASK_OVERLAY = True  # Toggle segmentation overlay
SCALE_FACTOR = 0.4  # More aggressive downscale for speed (0.4 = 40%)

# Crop and resize helper
def crop_and_resize(image, scale_factor=SCALE_FACTOR):
    x_min = 385
    x_max = 1708
    offset = 100

    x_min = 0
    x_max = image.shape[1]  # Use full width of the image

    height, width = image.shape[:2]
    x_start = max(0, x_min - offset)
    x_end = min(width, x_max + offset)

    cropped = image[:, x_start:x_end]
    resized = cv2.resize(cropped, (int(cropped.shape[1] * scale_factor), int(cropped.shape[0] * scale_factor)), interpolation=cv2.INTER_AREA)

    return cropped, resized, (x_start, x_end)

# Open video capture
file_path = r"C:\Users\rkjo\OneDrive\Documents\pool_tracking\VID20250310120030.mp4"
cap = cv2.VideoCapture(file_path)

# Load YOLOv8 model
model = YOLO('yolov8m-seg.pt')

# Initialize tracker only if needed
if USE_TRACKING:
    tracker = DeepSort(max_age=999999, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.2)

prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Step 1: Crop and resize
    cropped_frame, resized_frame, (x_start, x_end) = crop_and_resize(frame, SCALE_FACTOR)

    # Step 2: Run YOLOv8 inference
    results = model.predict(resized_frame, save=False, stream=False, verbose=False)
    result = results[0]

    detections = []
    boxes_for_draw = []
    masks_to_apply = []

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    if result.masks is not None:
        mask_data = result.masks.data.cpu().numpy()
    else:
        mask_data = None

    for idx, (box, score, cls_id) in enumerate(zip(boxes, scores, classes)):
        if int(cls_id) == 0:  # Only person class
            x1, y1, x2, y2 = box

            # Rescale box back to cropped frame size
            x1 = x1 / SCALE_FACTOR
            x2 = x2 / SCALE_FACTOR
            y1 = y1 / SCALE_FACTOR
            y2 = y2 / SCALE_FACTOR

            detections.append(([int(x1), int(y1), int(x2 - x1), int(y2 - y1)], float(score), int(cls_id)))
            boxes_for_draw.append((x1, y1, x2, y2))

            if mask_data is not None:
                masks_to_apply.append(mask_data[idx])

    if USE_TRACKING:
        # Update tracker
        tracks = tracker.update_tracks(detections, frame=cropped_frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, w, h = track.to_ltrb()

            cv2.rectangle(cropped_frame, (int(l), int(t)), (int(l + w), int(t + h)), (255, 0, 0), 2)
            cv2.putText(cropped_frame, f'ID: {track_id}', (int(l), int(t) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        for (x1, y1, x2, y2) in boxes_for_draw:
            cv2.rectangle(cropped_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Overlay only PERSON masks
    # Overlay only PERSON masks
    if USE_MASK_OVERLAY and masks_to_apply:
        # Initialize combined mask with resized_frame shape
        final_mask = np.zeros((resized_frame.shape[0], resized_frame.shape[1]), dtype=bool)

        for mask in masks_to_apply:
            resized_mask = cv2.resize(mask.astype('float32'), (resized_frame.shape[1], resized_frame.shape[0]))
            resized_mask = resized_mask > 0.5  # Threshold
            final_mask |= resized_mask  # Combine masks

        # Resize final_mask back to original cropped_frame size
        final_mask = cv2.resize(final_mask.astype('float32'), (cropped_frame.shape[1], cropped_frame.shape[0]))
        final_mask = final_mask > 0.5  # Threshold again

        color_mask = np.zeros_like(cropped_frame)
        color_mask[final_mask] = (0, 255, 0)
        cropped_frame = cv2.addWeighted(cropped_frame, 1.0, color_mask, 0.5, 0)

    # Show optimized window
    display_frame = cv2.resize(cropped_frame, (cropped_frame.shape[1]//2, cropped_frame.shape[0]//2))  # Optional display downscale
    cv2.imshow('Optimized View', display_frame)

    # FPS counter
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.setWindowTitle('Optimized View', f'Optimized View - {fps:.2f} FPS')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
