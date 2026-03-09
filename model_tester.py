import os
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from helper_classes.file_loader_class import FileLoader
from helper_classes.config_loader import ConfigLoader

#### MODEL FOLDER
output_folder = "output_2026_03_09_09_00" ## "output_2026_03_08_20_20" ## output_2026_03_07_23_43

#### REAL / SYNTHETIC DATA
use_real_test_data = True

### CONFIG-LOADER
main_folder = os.path.dirname(os.path.abspath(__file__)) + "/"
config = ConfigLoader(main_folder + "config_file.txt")

EXPORT_W = config.width
EXPORT_H = config.height

### Pfade
model_path = config.model_output_path + output_folder + "/model.keras"

if use_real_test_data:
    target_folder = config.test_data_real_path
else:
    target_folder = config.test_data_synthetic_path


# Diese Funktion MUSS vor dem Laden definiert sein
def my_preprocess(x, training):
    return keras.applications.mobilenet_v2.preprocess_input(x)


model = keras.models.load_model(
    model_path,
    custom_objects={
        "my_preprocess": my_preprocess
    }
)

#### INIT LOADER
loader = FileLoader(target_folder, EXPORT_W, EXPORT_H).load()
loader.load_label_data(config.model_output_path + output_folder)

#### FILE-DATA
images = loader.images
y_bbox = loader.y_bbox
y_class = loader.y_class
label_names = loader.label_names


print(f"\n{len(images)} Testbilder | Labels: {label_names}")


# --- Evaluate ---
results = model.evaluate(images, [y_class, y_bbox])
print(f"\nTest Loss:       {results[0]:.4f}")
print(f"Class Accuracy:  {results[3]:.4f}")
print(f"BBox R2 Score:   {results[4]:.4f}")


# --- Predict ---
pred_class, pred_bbox = model.predict(images)
pred_labels = np.argmax(pred_class, axis=1)

correct = np.sum(pred_labels == y_class)
print(f"\nKlassifikation: {correct}/{len(y_class)} richtig ({correct/len(y_class)*100:.1f}%)")
print(f"folder: {target_folder}")

# --- Visualisierung ---

PLOT_COLS = 5
PLOT_ROWS = 5
PLOT_COUNT = PLOT_COLS * PLOT_ROWS

correct_indices = np.where(pred_labels == y_class)[0]
wrong_indices = np.where(pred_labels != y_class)[0]
np.random.shuffle(correct_indices)
np.random.shuffle(wrong_indices)

for title, indices, color in [
    (f"RICHTIG — {len(correct_indices)}/{len(y_class)}", correct_indices[:PLOT_COUNT], 'green'),
    (f"FALSCH — {len(wrong_indices)}/{len(y_class)}", wrong_indices[:PLOT_COUNT], 'red'),
]:
    if len(indices) == 0:
        continue

    rows = int(np.ceil(len(indices) / PLOT_COLS))
    fig, axes = plt.subplots(rows, PLOT_COLS, figsize=(15, 4 * rows))

    # axes immer 2D
    if rows == 1:
        axes = [axes]

    for plot_idx, i in enumerate(indices):
        row = plot_idx // PLOT_COLS
        col = plot_idx % PLOT_COLS
        ax = axes[row][col]
        ax.imshow(images[i])

        # Vorhersage (grün)
        xc = pred_bbox[i][0] * EXPORT_W
        yc = pred_bbox[i][1] * EXPORT_H
        w  = pred_bbox[i][2] * EXPORT_W
        h  = pred_bbox[i][3] * EXPORT_H
        ax.add_patch(patches.Rectangle(
            (xc - w / 2, yc - h / 2), w, h, linewidth=2,
            edgecolor='lime', facecolor='none', linestyle='-'
        ))

        # Ground Truth (rot)
        xc = y_bbox[i][0] * EXPORT_W
        yc = y_bbox[i][1] * EXPORT_H
        w  = y_bbox[i][2] * EXPORT_W
        h  = y_bbox[i][3] * EXPORT_H
        ax.add_patch(patches.Rectangle(
            (xc - w / 2, yc - h / 2), w, h, linewidth=2,
            edgecolor='red', facecolor='none', linestyle='--'
        ))


        pred_name = label_names[pred_labels[i]][:15]
        true_name = label_names[y_class[i]][:15]

        pred_conf = pred_class[i][pred_labels[i]] * 100
        true_conf = pred_class[i][y_class[i]] * 100

        if pred_labels[i] == y_class[i]:
            ax.set_title(f"✓ {pred_name} ({pred_conf:.1f}%)", fontsize=12, color='green')
        else:
            ax.set_title(f"✗ {pred_name} ({pred_conf:.1f}%)\n→ {true_name} ({true_conf:.1f}%)", fontsize=12, color='red')

        ax.axis('off')

    # Leere Felder ausblenden
    for j in range(len(indices), rows * PLOT_COLS):
        axes[j // PLOT_COLS][j % PLOT_COLS].axis('off')

    plt.suptitle(f"{title} | Grün=Pred, Rot=GT", fontsize=14, color=color, y=0.99)
    plt.tight_layout()
    plt.show()

