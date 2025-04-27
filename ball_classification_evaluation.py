from pathlib import Path

import cv2
import matplotlib

matplotlib.use('Qt5Agg')  # Or 'TkAgg' if you don't have Qt
import matplotlib.pyplot as plt


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Import your classification functions
from ball_classification import classify_pool_ball_from_masked_image

# --- SETTINGS ---
GT_FOLDER = Path(r"C:\Users\rkjo\OneDrive\Documents\pool_tracking\ball_classification_ground_truth")
WRONG_PREDICTIONS_OUTPUT = Path(r"C:\Users\rkjo\OneDrive\Documents\pool_tracking\wrong_predictions2")
WRONG_PREDICTIONS_OUTPUT.mkdir(parents=True, exist_ok=True)

# --- HELPER FUNCTIONS ---
def get_simple_label(full_label):
    if full_label == "Cue_ball":
        return "Cue"
    if "Solid" in full_label:
        return "Solid"
    if "Striped" in full_label:
        return "Stripe"
    return "Unknown"

# --- EVALUATION ---
true_labels = []
predicted_labels = []

ctr = 0

for subfolder in sorted(GT_FOLDER.iterdir()):
    if not subfolder.is_dir():
        continue

    ground_truth_label = subfolder.name

    for image_path in subfolder.glob("*.png"):
        input_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        color_image = input_image[:, :, :3]
        mask = input_image[:, :, 3]

        prediction = classify_pool_ball_from_masked_image(color_image, mask)

        true_labels.append(ground_truth_label)
        predicted_labels.append(prediction)

        if prediction != ground_truth_label:
            # Save wrong prediction
            save_path = WRONG_PREDICTIONS_OUTPUT / f"GT_{ground_truth_label}__PRED_{prediction}__{image_path.name}"
            cv2.imwrite(str(save_path), input_image)

        ctr += 1
        if ctr % 200 == 0:
            print(f"Processed {ctr} images...")

print(f"Finished processing {ctr} images.")

# --- RESULTS ---
print("\n=== Classification Report (Full Labels) ===")
print(classification_report(true_labels, predicted_labels, zero_division=0))

# Full Confusion Matrix
labels_sorted = sorted(list(set(true_labels + predicted_labels)))
cm = confusion_matrix(true_labels, predicted_labels, labels=labels_sorted)
fig, ax = plt.subplots(figsize=(12, 12))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted).plot(ax=ax, cmap='Blues', xticks_rotation=90)
plt.title("Confusion Matrix - Full Labels")
plt.tight_layout()
plt.show()

# Simplified Solid/Stripe/Cue
true_simple = [get_simple_label(lbl) for lbl in true_labels]
pred_simple = [get_simple_label(lbl) for lbl in predicted_labels]

print("\n=== Classification Report (Solid/Stripe/Cue) ===")
print(classification_report(true_simple, pred_simple, zero_division=0))

simple_labels = ["Solid", "Stripe", "Cue"]
cm_simple = confusion_matrix(true_simple, pred_simple, labels=simple_labels)
fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay(confusion_matrix=cm_simple, display_labels=simple_labels).plot(ax=ax, cmap='Greens')
plt.title("Confusion Matrix - Solid vs Stripe vs Cue")
plt.tight_layout()
plt.show()
