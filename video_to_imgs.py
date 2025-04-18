import cv2
import os

def extract_frames(video_path, output_folder, image_format='png'):
    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Construct image filename
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.{image_format}")

        # Save frame as image
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to '{output_folder}'.")

# Example usage:
video_path = r"C:\Users\rkjo\OneDrive\Documents\pool_tracking\VID20250310120030.mp4"
output_folder = r'C:\Users\rkjo\OneDrive\Documents\pool_tracking\frames_output'
extract_frames(video_path, output_folder)
