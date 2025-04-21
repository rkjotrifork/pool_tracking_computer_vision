import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab
from pathlib import Path
from collections import defaultdict

from scipy.stats import gaussian_kde

# Define root path
ground_truth_root = Path(r"C:\Users\rkjo\OneDrive\Documents\pool_tracking\ball_classification_ground_truth")

# Known colors used in classification
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

def classify_folder(name):
    if "Cue" in name:
        return "Cue", "White"
    elif "Striped" in name:
        color = name.split("_")[-1].lower()
        return "Stripe", color
    else:
        color = name.split("_")[-1].lower()
        return "Solid", color

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

    white_pixels = lab_pixels[
        (lab_pixels[:, 0] > 80) &
        (np.abs(lab_pixels[:, 1]) < white_color_offset) &
        (np.abs(lab_pixels[:, 2])  < white_color_offset)
        ]
    white_ratio = len(white_pixels) / len(lab_pixels)

    color_pixels = lab_pixels[
        (np.abs(lab_pixels[:, 1]) > white_color_offset) |
        (np.abs(lab_pixels[:, 2]) > white_color_offset)
        ]
    avg_color = np.mean(color_pixels, axis=0) if len(color_pixels) > 0 else [0, 0, 0]

    return white_ratio, avg_color

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

    plt.tight_layout()
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