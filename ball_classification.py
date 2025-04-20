from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2lab


# Known CIE Lab reference colors for pool balls
ball_colors = {
    'yellow': [80, -5, 75],
    'blue': [40, 10, -45],
    'red': [50, 65, 45],
    'purple': [35, 45, -25],
    'orange': [60, 30, 70],
    'green': [50, -50, 50],
    'maroon': [35, 45, 30],
    'black': [20, 0, 0]
}


ball_label_map = {
    ("Solid", "Yellow"): "1_Solid_Yellow",
    ("Solid", "Blue"): "2_Solid_Blue",
    ("Solid", "Red"): "3_Solid_Red",
    ("Solid", "Purple"): "4_Solid_Purple",
    ("Solid", "Orange"): "5_Solid_Orange",
    ("Solid", "Green"): "6_Solid_Green",
    ("Solid", "Maroon"): "7_Solid_Maroon",
    ("Solid", "Black"): "8_Solid_Black",
    ("Stripe", "Yellow"): "9_Striped_Yellow",
    ("Stripe", "Blue"): "10_Striped_Blue",
    ("Stripe", "Red"): "11_Striped_Red",
    ("Stripe", "Purple"): "12_Striped_Purple",
    ("Stripe", "Orange"): "13_Striped_Orange",
    ("Stripe", "Green"): "14_Striped_Green",
    ("Stripe", "Maroon"): "15_Striped_Maroon",
    ("Cue", "White"): "Cue_ball"
}


def closest_color(lab_val):
    min_dist = float('inf')
    closest_name = None
    for name, ref in ball_colors.items():
        dist = np.linalg.norm(np.array(lab_val) - np.array(ref))
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name

def classify_pool_ball_from_masked_image(image_bgr, mask):
    # Extract only masked pixels
    masked_pixels = image_bgr[mask > 0]

    if masked_pixels.size == 0:
        return "No ball pixels detected"

    # Convert to RGB then to Lab
    masked_rgb = cv2.cvtColor(masked_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
    lab_pixels = rgb2lab(masked_rgb.reshape(1, -1, 3)).reshape(-1, 3)

    # Count white pixels
    white_pixels = lab_pixels[(np.abs(lab_pixels[:, 1]) < 5) & (np.abs(lab_pixels[:, 2]) < 5)]
    white_ratio = len(white_pixels) / len(lab_pixels)

    if white_ratio > 0.9:
        return "Cue Ball (white)"

    is_stripe = white_ratio > 0.3

    # Exclude white pixels to find dominant color
    color_pixels = lab_pixels[(np.abs(lab_pixels[:, 1]) > 5) | (np.abs(lab_pixels[:, 2]) > 5)]

    if len(color_pixels) == 0:
        return "Unknown"

    avg_color = np.mean(color_pixels, axis=0)
    main_color = closest_color(avg_color)

    if white_ratio > 0.9:
        return ball_label_map[("Cue", "White")]

    ball_type = "Stripe" if is_stripe else "Solid"
    ball_key = (ball_type, main_color.capitalize())

    return ball_label_map.get(ball_key, f"Unknown_{ball_type}_{main_color}")



if __name__ == "__main__":

    input_folder_path = Path(r"C:\Users\rkjo\OneDrive\Documents\pool_tracking\circles_output")

    output_folder_path = Path(r"C:\Users\rkjo\OneDrive\Documents\pool_tracking\ball_classification_results")

    ctr = 0
    for image_path in input_folder_path.glob("*.png"):
        input_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        color_image = input_image[:, :, :3]
        mask = input_image[:, :, 3]  # Assuming the mask is in the alpha channel

        result = classify_pool_ball_from_masked_image(color_image, mask)

        #save input image in output folder and then subfolder with the result name
        result_folder = output_folder_path / result.replace(" ", "_")
        result_folder.mkdir(parents=True, exist_ok=True)
        output_image_path = result_folder / image_path.name
        cv2.imwrite(str(output_image_path), input_image)

        img = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(result)
        plt.axis('off')
        plt.show()


        ctr += 1
        if ctr % 10 == 0:
            print(f"Processed {ctr} images.")
            break