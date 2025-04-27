from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from skimage.color import deltaE_ciede2000
from skimage.color import rgb2lab, lab2rgb
from sklearn.cluster import KMeans

from ball_analysis import calculate_white_ratio

# Known CIE Lab reference colors for pool balls
# ball_colors = {
#     'yellow': [80, -5, 75],
#     'blue': [40, 10, -45],
#     'red': [50, 65, 45],
#     'purple': [35, 45, -25],
#     'orange': [60, 30, 70],
#     'green': [50, -50, 50],
#     'maroon': [35, 45, 30],
#     'black': [20, 0, 0]
# }


ball_colors = {
    # 'White': [93, -3, 18],
    'black': [46, 4, 13],
    'blue': [55, 13, -38],
    'green': [69, -33, 23],
    'maroon': [51, 37, 19],
    'orange': [78, 21, 31],
    'purple': [49, 31, -36],
    'red': [61, 54, 31],
    'yellow': [84, 1, 64],
}

ball_colors= {
    # 'White': [92.2, -1.4, 18.3],
    'black': [45.0, 4.0, 13.8],
    'blue': [49.2, 15.8, -45.0],
    'green': [67.1, -32.3, 24.4],
    'maroon': [47.8, 41.1, 18.2],
    'orange': [81.0, 19.5, 26.4],
    'purple': [47.5, 33.9, -38.1],
    'red': [60.4, 56.9, 30.3],
    'yellow': [83.3, 0.5, 68.3],
}

ball_colors = {
'black': [54.1, 6.8, 20.5],
'blue': [49.0, 17.0, -48.6],
'green': [65.0, -34.8, 24.7],
'maroon': [47.6, 42.5, 18.3],
'orange': [77.7, 24.8, 29.9],
'purple': [47.4, 34.7, -43.1],
'red': [59.9, 57.6, 30.5],
'yellow': [82.6, 0.5, 67.6],
}

ball_colors = {
    # 'White': [90.9, -0.6, 20.8],
    'black': [53.8, 6.8, 20.5],
    'blue': [48.4, 17.5, -49.0],
    'green': [65.5, -40.1, 17.9],
    'maroon': [47.8, 42.2, 18.3],
    'orange': [74.3, 25.3, 39.3],
    'purple': [48.8, 34.9, -35.7],
    'red': [59.9, 57.9, 30.5],
    'yellow': [82.4, 0.6, 68.2],
}

#same as ball_colors but without black (make sure it is a dict)
ball_colors_stripe = ball_colors.copy()
ball_colors_stripe.pop('black')


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


def get_color_from_histogram_peak(color_pixels):
    # get peak color from histogram
    peaks = []
    for i in range(3):  # For L*, a*, b*
        values = color_pixels[:, i]
        kde = gaussian_kde(values)

        # Create a fine grid to evaluate KDE over
        x_grid = np.linspace(values.min(), values.max(), 500)
        kde_values = kde(x_grid)

        # Find mode (maximum of KDE)
        peak_val = x_grid[np.argmax(kde_values)]
        peaks.append(round(peak_val, 1))
    color = peaks
    return color


def find_color_with_kmeans2(lab_pixels, image_mask, n_clusters=3):
    lab_pixels_reshaped = lab_pixels.reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(lab_pixels_reshaped)
    cluster_labels = kmeans.labels_
    cluster_colors = kmeans.cluster_centers_
    cluster_labels_flat = cluster_labels.astype(int)

    # Reconstruct a full-size LAB image using the mask
    quantized_lab = np.zeros((*image_mask.shape, 3), dtype=np.float32)
    mask_indices = np.argwhere(image_mask > 0)

    for idx, (y, x) in enumerate(mask_indices):
        quantized_lab[y, x] = cluster_colors[cluster_labels_flat[idx]]

    # Convert to RGB and show

    # Identify white clusters
    white_offset = 20
    white_mask = np.array([
        (c[0] > 85 and abs(c[1]) < white_offset and abs(c[2]) < white_offset)
        for c in cluster_colors
    ])

    white_indices = np.where(white_mask)[0]
    non_white_indices = np.where(~white_mask)[0]

    # Choose dominant non-white cluster if large enough
    min_cluster_size = 0.05 * len(lab_pixels)
    cluster_sizes = np.bincount(cluster_labels)

    print(f"Non-white clusters: {cluster_colors[non_white_indices]}")


    if len(non_white_indices) > 0:
        largest_non_white_idx = non_white_indices[np.argmax(cluster_sizes[non_white_indices])]
        if cluster_sizes[largest_non_white_idx] > min_cluster_size:
            return cluster_colors[largest_non_white_idx]

    # Fallback to largest cluster (likely white)
    largest_idx = np.argmax(cluster_sizes)
    return cluster_colors[largest_idx]


def find_color_with_kmeans(lab_pixels, image_mask, is_stripe=False, n_clusters=3):
    lab_pixels_reshaped = lab_pixels.reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(lab_pixels_reshaped)
    cluster_labels = kmeans.labels_
    cluster_colors = kmeans.cluster_centers_
    cluster_labels_flat = cluster_labels.astype(int)

    # Reconstruct a full-size LAB image using the mask
    quantized_lab = np.zeros((*image_mask.shape, 3), dtype=np.float32)
    mask_indices = np.argwhere(image_mask > 0)

    for idx, (y, x) in enumerate(mask_indices):
        quantized_lab[y, x] = cluster_colors[cluster_labels_flat[idx]]

    # Show quantized LAB image
    plotting = False
    if plotting:
        quantized_rgb = lab2rgb(quantized_lab)
        plt.imshow((quantized_rgb * 255).astype(np.uint8))
        plt.axis('off')
        plt.title("Quantized LAB Image (KMeans)")
        plt.show()

    # Define thresholds
    white_offset = 20
    min_cluster_size = 0.05 * len(lab_pixels)

    # Determine valid clusters
    cluster_sizes = np.bincount(cluster_labels)
    valid_clusters = []
    for i, color in enumerate(cluster_colors):
        is_white = (color[0] > 85 and abs(color[1]) < white_offset and abs(color[2]) < white_offset)
        if not is_white and cluster_sizes[i] > min_cluster_size:
            valid_clusters.append((i, color))

    if not valid_clusters:
        # Fallback to largest overall cluster
        largest_idx = np.argmax(cluster_sizes)
        return cluster_colors[largest_idx]

    # Match each valid cluster to the closest known color
    color_map = ball_colors_stripe if is_stripe else ball_colors
    best_match = None
    best_distance = float('inf')

    for i, cluster_color in valid_clusters:
        cluster_lab = np.array(cluster_color).reshape(1, 1, 3)
        for name, ref_lab in color_map.items():
            ref_lab_arr = np.array(ref_lab).reshape(1, 1, 3)
            dist = deltaE_ciede2000(cluster_lab, ref_lab_arr)[0][0]
            if dist < best_distance:
                best_distance = dist
                best_match = cluster_color

    return best_match


def closest_color(lab_val, is_stripe):

    # colors = ball_colors if not is_stripe else ball_colors_stripe
    colors = ball_colors

    min_dist = float('inf')
    closest_name = None
    for name, ref in colors.items():
        dist = np.linalg.norm(np.array(lab_val) - np.array(ref))
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name

def closest_color2(lab_val, is_stripe):
    colors = ball_colors if not is_stripe else ball_colors_stripe

    min_dist = float('inf')
    closest_name = None

    lab_val = np.array(lab_val).reshape(1, 1, 3)  # 3D shape needed for deltaE_ciede2000

    for name, ref in colors.items():
        ref_lab = np.array(ref).reshape(1, 1, 3)
        dist = deltaE_ciede2000(lab_val, ref_lab)[0][0]
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name

def classify_pool_ball_from_masked_image(image_bgr, mask):

    #add 1 pixel of padding to mask and image
    mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    image_bgr = cv2.copyMakeBorder(image_bgr, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    #erode mask to remove noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)

    # Extract only masked pixels
    masked_pixels = image_bgr[mask > 0]

    if masked_pixels.size == 0:
        return "No ball pixels detected"

    # Convert to RGB then to Lab
    masked_rgb = cv2.cvtColor(masked_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
    lab_pixels = rgb2lab(masked_rgb.reshape(1, -1, 3)).reshape(-1, 3)

    # white_ratio = calculate_white_ratio_old(lab_pixels)
    white_ratio = calculate_white_ratio(lab_pixels, mask, plotting=False)

    # if white_ratio > 0.5:
    if white_ratio > 0.95:
        # return "Cue Ball (white)"
        return ball_label_map[("Cue", "White")]

    is_stripe = white_ratio > 0.33
    # is_stripe = white_ratio > 0.1

    # Exclude white pixels to find dominant color
    # color_pixels = lab_pixels[(np.abs(lab_pixels[:, 1]) > 5) | (np.abs(lab_pixels[:, 2]) > 5)]
    white_color_offset2=20

    color_pixels = lab_pixels[ # still including black
        (lab_pixels [:, 0] < 50) |
        (np.abs(lab_pixels[:, 1]) > white_color_offset2) |
        (np.abs(lab_pixels[:, 2]) > white_color_offset2)
        ]

    if len(color_pixels) == 0:
        return "Unknown"

    # avg_color = np.mean(color_pixels, axis=0)

    # color = get_color_from_histogram_peak(color_pixels)
    # color = find_color_with_kmeans(lab_pixels, mask)
    color = find_color_with_kmeans(lab_pixels, mask, is_stripe=is_stripe)

    main_color = closest_color(color, is_stripe) # TODO find color above also does dist calc. Make sure only done once.

    ball_type = "Stripe" if is_stripe else "Solid"

    if main_color == "black":
        # If the color is black, classify it as a solid ball
        ball_type = "Solid"

    ball_key = (ball_type, main_color.capitalize())

    return ball_label_map.get(ball_key, f"Unknown_{ball_type}_{main_color}")


def calculate_white_ratio_old(lab_pixels):
    # Count white pixels
    white_color_offset = 15
    # white_pixels = lab_pixels[(np.abs(lab_pixels[:, 1]) < 5) & (np.abs(lab_pixels[:, 2]) < 5)]
    white_pixels = lab_pixels[
        (lab_pixels[:, 0] > 80) &
        (np.abs(lab_pixels[:, 1]) < white_color_offset) &
        (np.abs(lab_pixels[:, 2]) < white_color_offset)
        ]
    white_ratio = len(white_pixels) / len(lab_pixels)
    return white_ratio


if __name__ == "__main__":

    input_folder_path = Path(r"C:\Users\rkjo\OneDrive\Documents\pool_tracking\circles_output")
    # input_folder_path = Path(r"C:\Users\rkjo\OneDrive\Documents\pool_tracking\ball_classification_results14\13_Striped_Orange")

    # input_folder_path = Path(r"C:\Users\rkjo\OneDrive\Documents\pool_tracking\wrong_predictions2")

    output_folder_path = Path(r"C:\Users\rkjo\OneDrive\Documents\pool_tracking\ball_classification_results_week17_9")

    ctr = 0
    for image_path in input_folder_path.glob("*.png"):
        print()
        print(f"Processing {image_path.name}...")
        input_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

        color_image = input_image[:, :, :3]
        mask = input_image[:, :, 3]  # Assuming the mask is in the alpha channel

        result = classify_pool_ball_from_masked_image(color_image, mask)

        #save input image in output folder and then subfolder with the result name
        result_folder = output_folder_path / result.replace(" ", "_")
        result_folder.mkdir(parents=True, exist_ok=True)
        output_image_path = result_folder / image_path.name
        cv2.imwrite(str(output_image_path), input_image)

        plotting = False
        if plotting:
            img = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.title(result)
            plt.axis('off')
            plt.show()


        ctr += 1
        if ctr % 200 == 0:
            print(f"Processed {ctr} images.")
            break