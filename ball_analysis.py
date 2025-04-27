from collections import defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from scipy.stats import gaussian_kde
from skimage.color import rgb2lab, lab2rgb

# matplotlib.use('TkAgg')  # Force use of TkAgg backend (opens a real window)
import matplotlib.pyplot as plt

from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans





# Function to compute histogram overlap
def histogram_overlap_counts(hist1, hist2):
    min_counts = np.minimum(hist1, hist2)
    total_overlap = np.sum(min_counts)
    return total_overlap, min_counts


def classify_folder(name):
    if "Cue" in name:
        return "Cue", "White"
    elif "Striped" in name:
        color = name.split("_")[-1].lower()
        return "Stripe", color
    else:
        color = name.split("_")[-1].lower()
        return "Solid", color



def calculate_white_ratio(lab_pixels, image_mask=None, plotting=False):

    # # --- New: White ratio estimation using KMeans on L ---
    # # L_values = lab_pixels[:, 0].reshape(-1, 1)
    # lab_values = lab_pixels[:, 1:3].reshape(-1, 2)
    # kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(lab_values)
    # labels = kmeans.labels_
    # cluster_centers = kmeans.cluster_centers_.flatten()


    # White ratio estimation using KMeans on LAB pixels
    lab_values = lab_pixels
    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(lab_values)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_


    # Identify the bright cluster (higher L value)
    cluster_centers_L_values = [center[0] for center in cluster_centers]

    # Identify the bright cluster (higher L value)
    bright_cluster_idx = np.argmax(cluster_centers_L_values)

    # Compute the size of the bright cluster
    bright_cluster_size = np.sum(labels == bright_cluster_idx)
    total_pixels = len(labels)
    white_ratio = bright_cluster_size / total_pixels

    # largest_cluster_idx = np.argmax(np.bincount(labels))
    # smallest_cluster_idx = np.argmin(np.bincount(labels))
    # outlier_small_cluster_size_threshold = 0.10 # TODO maybe not do the removal of small clusters
    # smallest_cluster_size = np.min(np.bincount(labels))
    # if smallest_cluster_size < outlier_small_cluster_size_threshold * total_pixels:
    #     #other cluster is too small to be meaningful, so set smallest cluster to same as largest
    #     smallest_cluster_idx = largest_cluster_idx


    # Merge clusters if they're very close in LAB space
    cluster_0_lab = np.mean(lab_pixels[labels == 0], axis=0)
    cluster_1_lab = np.mean(lab_pixels[labels == 1], axis=0)
    lab_distance = euclidean(cluster_0_lab, cluster_1_lab)

    WHITE_LAB = np.array([100, 0, 0])
    bright_cluster_lab = np.mean(lab_pixels[labels == bright_cluster_idx], axis=0)
    distance_to_white_bright = euclidean(bright_cluster_lab, WHITE_LAB)
    distance_to_white_0 = euclidean(cluster_0_lab, WHITE_LAB)
    distance_to_white_1 = euclidean(cluster_1_lab, WHITE_LAB)


    if plotting:
        if image_mask is None:
            raise ValueError("Image mask is required for plotting.")

        # Reconstruct full-size quantized LAB image
        quantized_lab = np.zeros((*image_mask.shape, 3), dtype=np.float32)
        mask_indices = np.argwhere(image_mask > 0)

        # Precompute cluster average LABs
        cluster_avg_lab = [
            np.mean(lab_pixels[labels == i], axis=0) for i in range(2)
        ]

        for idx, (y, x) in enumerate(mask_indices):
            quantized_lab[y, x] = cluster_avg_lab[labels[idx]]

        quantized_rgb = lab2rgb(quantized_lab)

        # Reconstruct original LAB image
        original_lab = np.zeros((*image_mask.shape, 3), dtype=np.float32)
        original_lab[image_mask > 0] = lab_pixels
        original_rgb = lab2rgb(original_lab)

        # Plot original
        plt.figure(figsize=(6,6))
        plt.imshow((original_rgb * 255).astype(np.uint8))
        plt.axis('off')
        plt.title("Original LAB Image (expanded from mask)")
        plt.tight_layout()
        plt.show()

        # Plot quantized
        plt.figure(figsize=(6,6))
        plt.imshow((quantized_rgb * 255).astype(np.uint8))
        plt.axis('off')
        plt.title("Quantized LAB Image (KMeans), White Ratio: {:.2f}, Cluster Dist: {:.2f}".format(white_ratio, lab_distance))
        plt.tight_layout()
        plt.show()


    BRIGHT_NOT_WHITE_THRESHOLD = 40  # Tunable threshold
    cluster_distance_threshold = 10  # Tunable threshold

    no_white_on_ball = distance_to_white_0 > BRIGHT_NOT_WHITE_THRESHOLD and distance_to_white_1 > BRIGHT_NOT_WHITE_THRESHOLD
    cluster_colors_close_to_each_other = lab_distance < cluster_distance_threshold

    if distance_to_white_bright > BRIGHT_NOT_WHITE_THRESHOLD:
        # if cluster_colors_close_to_each_other:
        #     # Clusters are close to each other, so we can merge them
        return 0.0

    if no_white_on_ball:
        # Both clusters are far from white
        return 0.0

    only_white_on_ball = distance_to_white_0 < BRIGHT_NOT_WHITE_THRESHOLD and distance_to_white_1 < BRIGHT_NOT_WHITE_THRESHOLD
    if only_white_on_ball:
        # Both clusters are close to white
        return 1.0

    return white_ratio


def extract_features_from_image(image_path, white_color_offset=15, b_bias=0):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if image.shape[2] < 4:
        return None, None

    color_image = image[:, :, :3]
    mask = image[:, :, 3]

    masked_pixels = color_image[mask > 0]
    if masked_pixels.size == 0:
        return None, None

    rgb_pixels = cv2.cvtColor(masked_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
    lab_pixels = rgb2lab(rgb_pixels.reshape(1, -1, 3)).reshape(-1, 3)

    # Shift b* values so that cue ball white is centered around 0
    lab_pixels[:, 2] = lab_pixels[:, 2] - b_bias

    # white_pixels = lab_pixels[
    #     (lab_pixels[:, 0] > 80) &
    #     (np.abs(lab_pixels[:, 1]) < white_color_offset) &
    #     (np.abs(lab_pixels[:, 2])  < white_color_offset)
    #     ]
    # white_ratio = len(white_pixels) / len(lab_pixels)

    white_ratio = calculate_white_ratio(lab_pixels, mask, plotting=False)

    color_pixels = lab_pixels[
        (np.abs(lab_pixels[:, 1]) > white_color_offset) |
        (np.abs(lab_pixels[:, 2]) > white_color_offset)
        ]
    avg_color = np.mean(color_pixels, axis=0) if len(color_pixels) > 0 else [0, 0, 0]

    return white_ratio, avg_color



if __name__ == "__main__":


    # Define root path
    ground_truth_root = Path(r"C:\Users\rkjo\OneDrive\Documents\pool_tracking\ball_classification_ground_truth")



    # Storage for global and color-specific stats
    global_stats = defaultdict(lambda: {"white_ratios": [], "colors": []})
    color_stats = defaultdict(lambda: {"white_ratios": [], "colors": []})

    # Traverse all folders
    for folder in ground_truth_root.iterdir():
        if not folder.is_dir():
            continue

        ball_type, color = classify_folder(folder.name)

        for image_path in folder.glob("*.png"):
            white_ratio, avg_color = extract_features_from_image(image_path)
            if white_ratio is None:
                continue

            global_stats[ball_type]["white_ratios"].append(white_ratio)
            global_stats[ball_type]["colors"].append(avg_color)

            if color != "white":  # skip cue ball for color stats
                color_stats[color]["white_ratios"].append(white_ratio)
                color_stats[color]["colors"].append(avg_color)


    # Compute histograms for each category
    histograms = {}
    bins = np.linspace(0, 1, 11)  # Same binning as used in plotting (for white_ratios)

    # === GLOBAL STATS BY TYPE ===
    for category in global_stats:
        print(f"\n=== {category.upper()} BALLS ===")
        white_ratios = np.array(global_stats[category]["white_ratios"])
        colors = np.array(global_stats[category]["colors"])

        hist, _ = np.histogram(white_ratios, bins=bins, density=False)  # Counts, not densities
        histograms[category] = hist

        print(f"White Ratio: Mean = {white_ratios.mean():.3f}, Std = {white_ratios.std():.3f}")
        if len(colors) > 0:
            print(f"Color L*: Mean = {colors[:,0].mean():.1f}, Std = {colors[:,0].std():.1f}")
            print(f"Color a*: Mean = {colors[:,1].mean():.1f}, Std = {colors[:,1].std():.1f}")
            print(f"Color b*: Mean = {colors[:,2].mean():.1f}, Std = {colors[:,2].std():.1f}")

        # Plot histograms
        plt.figure(figsize=(12, 4))
        plt.suptitle(f"{category} Ball Histogram Stats")

        plt.subplot(1, 4, 1)
        plt.hist(white_ratios, bins=10, color="lightgray", edgecolor="black")
        plt.title("White Ratio")

        if len(colors) > 0:
            plt.subplot(1, 4, 2)
            plt.hist(colors[:, 0], bins=10, color="lightblue", edgecolor="black")
            plt.title("L*")

            plt.subplot(1, 4, 3)
            plt.hist(colors[:, 1], bins=10, color="lightgreen", edgecolor="black")
            plt.title("a*")

            plt.subplot(1, 4, 4)
            plt.hist(colors[:, 2], bins=10, color="salmon", edgecolor="black")
            plt.title("b*")

        # plt.tight_layout()
        # plt.tight_layout(rect=[0, 0, 1, 0.85])
        # plt.suptitle(f"{category.capitalize()} Ball Histogram Stats", y=1.05)  # y > 1 moves it higher
        plt.show()

    # Compute and print pairwise overlaps
    print("\n=== Pairwise Histogram Overlaps (White Ratios) ===")
    for (cat1, cat2) in combinations(histograms.keys(), 2):
        overlap_total, min_counts = histogram_overlap_counts(histograms[cat1], histograms[cat2])
        print(f"\nOverlap between {cat1} and {cat2}: {overlap_total} total counts")
        print(f"Bin-wise minimum counts: {min_counts}")

    # === COLOR-SPECIFIC STATS ===
    print("\n\n====================")
    print("=== PER-COLOR ANALYSIS ===")
    print("====================")

    for color in color_stats:
        white_ratios = np.array(color_stats[color]["white_ratios"])
        colors = np.array(color_stats[color]["colors"])

        print(f"\n--- {color.upper()} ---")
        print(f"Samples: {len(white_ratios)}")
        print(f"White Ratio: Mean = {white_ratios.mean():.3f}, Std = {white_ratios.std():.3f}")
        print(f"Lab Color Mean = [{colors[:,0].mean():.1f}, {colors[:,1].mean():.1f}, {colors[:,2].mean():.1f}]")
        print(f"Lab Color Std  = [{colors[:,0].std():.1f}, {colors[:,1].std():.1f}, {colors[:,2].std():.1f}]")

        # Plot
        plt.figure(figsize=(10, 3))
        plt.suptitle(f"{color.capitalize()} Ball Histogram Stats")

        plt.subplot(1, 4, 1)
        plt.hist(white_ratios, bins=10, color="lightgray", edgecolor="black")
        plt.title("White Ratio")

        plt.subplot(1, 4, 2)
        plt.hist(colors[:, 0], bins=10, color="lightblue", edgecolor="black")
        plt.title("L*")

        plt.subplot(1, 4, 3)
        plt.hist(colors[:, 1], bins=10, color="lightgreen", edgecolor="black")
        plt.title("a*")

        plt.subplot(1, 4, 4)
        plt.hist(colors[:, 2], bins=10, color="salmon", edgecolor="black")
        plt.title("b*")

        plt.tight_layout(rect=[0, 0, 1, 0.85])
        plt.show()


    # === COMPILED COLOR VALUES FOR CLASSIFIER ===
    print("\n\n====================")
    print("=== FINAL OUTPUT FOR CLASSIFIER ===")
    print("====================")

    # Generate updated ball_colors based on mean
    print("\nball_colors = {")
    for color in sorted(color_stats.keys()):
        colors = np.array(color_stats[color]["colors"])
        if len(colors) > 0:
            mean_lab = colors.mean(axis=0)
            print(f"    '{color}': [{mean_lab[0]:.0f}, {mean_lab[1]:.0f}, {mean_lab[2]:.0f}],")
    print("}")

    # Print histogram peak values (mode-like estimates)
    print("\nball_color_peaks = {")
    for color in sorted(color_stats.keys()):
        colors = np.array(color_stats[color]["colors"])
        if len(colors) > 0:
            peaks = []
            for i in range(3):  # For L*, a*, b*
                values = colors[:, i]
                kde = gaussian_kde(values)

                # Create a fine grid to evaluate KDE over
                x_grid = np.linspace(values.min(), values.max(), 500)
                kde_values = kde(x_grid)

                # Find mode (maximum of KDE)
                peak_val = x_grid[np.argmax(kde_values)]
                peaks.append(round(peak_val, 1))

            print(f"    '{color}': [{peaks[0]}, {peaks[1]}, {peaks[2]}],")
    print("}")


    # Print white ratio thresholds for cue and stripe
    if "Cue" in global_stats:
        cue_white_ratios = np.array(global_stats["Cue"]["white_ratios"])
        cue_peak_bin = np.histogram(cue_white_ratios, bins=20)
        peak_index = np.argmax(cue_peak_bin[0])
        cue_peak_center = (cue_peak_bin[1][peak_index] + cue_peak_bin[1][peak_index + 1]) / 2
        print(f"\nCue Ball White Ratio: Mean = {cue_white_ratios.mean():.3f}, Std = {cue_white_ratios.std():.3f}, Peak ≈ {cue_peak_center:.3f}")

    if "Stripe" in global_stats:
        stripe_white_ratios = np.array(global_stats["Stripe"]["white_ratios"])
        stripe_peak_bin = np.histogram(stripe_white_ratios, bins=20)
        peak_index = np.argmax(stripe_peak_bin[0])
        stripe_peak_center = (stripe_peak_bin[1][peak_index] + stripe_peak_bin[1][peak_index + 1]) / 2
        print(f"Striped Ball White Ratio: Mean = {stripe_white_ratios.mean():.3f}, Std = {stripe_white_ratios.std():.3f}, Peak ≈ {stripe_peak_center:.3f}")

    if "Solid" in global_stats:
        solid_white_ratios = np.array(global_stats["Solid"]["white_ratios"])
        solid_peak_bin = np.histogram(solid_white_ratios, bins=20)
        peak_index = np.argmax(solid_peak_bin[0])
        solid_peak_center = (solid_peak_bin[1][peak_index] + solid_peak_bin[1][peak_index + 1]) / 2
        print(f"Solid Ball White Ratio: Mean = {solid_white_ratios.mean():.3f}, Std = {solid_white_ratios.std():.3f}, Peak ≈ {solid_peak_center:.3f}")




    # Traverse all folders
    for folder in ground_truth_root.iterdir():
        if not folder.is_dir():
            continue

        ball_type, color = classify_folder(folder.name)

        if ball_type != "Solid":
            continue  # Skip non-solid balls entirely for color analysis

        for image_path in folder.glob("*.png"):
            white_ratio, avg_color = extract_features_from_image(image_path)
            if white_ratio is None:
                continue

            global_stats[ball_type]["white_ratios"].append(white_ratio)
            global_stats[ball_type]["colors"].append(avg_color)

            color_stats[color]["white_ratios"].append(white_ratio)
            color_stats[color]["colors"].append(avg_color)

    # === GLOBAL STATS BY TYPE ===
    for category in global_stats:
        print(f"\n=== {category.upper()} BALLS ===")
        white_ratios = np.array(global_stats[category]["white_ratios"])
        colors = np.array(global_stats[category]["colors"])

        print(f"White Ratio: Mean = {white_ratios.mean():.3f}, Std = {white_ratios.std():.3f}")
        if len(colors) > 0:
            print(f"Color L*: Mean = {colors[:,0].mean():.1f}, Std = {colors[:,0].std():.1f}")
            print(f"Color a*: Mean = {colors[:,1].mean():.1f}, Std = {colors[:,1].std():.1f}")
            print(f"Color b*: Mean = {colors[:,2].mean():.1f}, Std = {colors[:,2].std():.1f}")

        # Plot histograms
        plt.figure(figsize=(12, 4))
        plt.suptitle(f"{category} Ball Histogram Stats")

        plt.subplot(1, 4, 1)
        plt.hist(white_ratios, bins=10, color="lightgray", edgecolor="black")
        plt.title("White Ratio")

        if len(colors) > 0:
            plt.subplot(1, 4, 2)
            plt.hist(colors[:, 0], bins=10, color="lightblue", edgecolor="black")
            plt.title("L*")

            plt.subplot(1, 4, 3)
            plt.hist(colors[:, 1], bins=10, color="lightgreen", edgecolor="black")
            plt.title("a*")

            plt.subplot(1, 4, 4)
            plt.hist(colors[:, 2], bins=10, color="salmon", edgecolor="black")
            plt.title("b*")

        plt.tight_layout()
        plt.show()

    # === COLOR-SPECIFIC STATS ===
    print("\n\n====================")
    print("=== PER-COLOR ANALYSIS (SOLIDS ONLY) ===")
    print("====================")

    for color in color_stats:
        white_ratios = np.array(color_stats[color]["white_ratios"])
        colors = np.array(color_stats[color]["colors"])

        print(f"\n--- {color.upper()} ---")
        print(f"Samples: {len(white_ratios)}")
        print(f"White Ratio: Mean = {white_ratios.mean():.3f}, Std = {white_ratios.std():.3f}")
        print(f"Lab Color Mean = [{colors[:,0].mean():.1f}, {colors[:,1].mean():.1f}, {colors[:,2].mean():.1f}]")
        print(f"Lab Color Std  = [{colors[:,0].std():.1f}, {colors[:,1].std():.1f}, {colors[:,2].std():.1f}]")

        # Plot
        plt.figure(figsize=(10, 3))
        plt.suptitle(f"{color.capitalize()} Ball Histogram Stats")

        plt.subplot(1, 4, 1)
        plt.hist(white_ratios, bins=10, color="lightgray", edgecolor="black")
        plt.title("White Ratio")

        plt.subplot(1, 4, 2)
        plt.hist(colors[:, 0], bins=10, color="lightblue", edgecolor="black")
        plt.title("L*")

        plt.subplot(1, 4, 3)
        plt.hist(colors[:, 1], bins=10, color="lightgreen", edgecolor="black")
        plt.title("a*")

        plt.subplot(1, 4, 4)
        plt.hist(colors[:, 2], bins=10, color="salmon", edgecolor="black")
        plt.title("b*")

        plt.tight_layout()
        plt.show()

    # === FINAL OUTPUT FOR CLASSIFIER (SOLIDS ONLY) ===
    print("\n\n====================")
    print("=== FINAL OUTPUT FOR CLASSIFIER (SOLIDS ONLY) ===")
    print("====================")

    # Generate updated ball_colors based on mean
    print("\nball_colors = {")
    for color in sorted(color_stats.keys()):
        colors = np.array(color_stats[color]["colors"])
        if len(colors) > 0:
            mean_lab = colors.mean(axis=0)
            print(f"    '{color}': [{mean_lab[0]:.0f}, {mean_lab[1]:.0f}, {mean_lab[2]:.0f}],")
    print("}")

    # Print histogram peak values (mode-like estimates)
    print("\nball_color_peaks = {")
    for color in sorted(color_stats.keys()):
        colors = np.array(color_stats[color]["colors"])
        if len(colors) > 0:
            peaks = []
            for i in range(3):  # For L*, a*, b*
                values = colors[:, i]
                kde = gaussian_kde(values)
                x_grid = np.linspace(values.min(), values.max(), 500)
                kde_values = kde(x_grid)
                peak_val = x_grid[np.argmax(kde_values)]
                peaks.append(round(peak_val, 1))

            print(f"    '{color}': [{peaks[0]}, {peaks[1]}, {peaks[2]}],")
    print("}")

    # Print white ratio threshold for solids only
    if "Solid" in global_stats:
        solid_white_ratios = np.array(global_stats["Solid"]["white_ratios"])
        solid_peak_bin = np.histogram(solid_white_ratios, bins=20)
        peak_index = np.argmax(solid_peak_bin[0])
        solid_peak_center = (solid_peak_bin[1][peak_index] + solid_peak_bin[1][peak_index + 1]) / 2
        print(f"\nSolid Ball White Ratio: Mean = {solid_white_ratios.mean():.3f}, Std = {solid_white_ratios.std():.3f}, Peak ≈ {solid_peak_center:.3f}")