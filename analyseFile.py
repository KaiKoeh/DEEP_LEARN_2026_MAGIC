import os
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from file_loader_class import FileLoader

pic_target_resolution = 256

### Pfade
##model_path = "mtg_detector.keras"

model_path = "mtg_detector_synthetic.keras"
##target_folder = "/Users/kaikohrsen/Documents/schulung/PythonWeekly/deep_learn_project/final_data/test_data_real"
##target_folder = r"C:\Users\MrKoiKoi\PycharmProjects\PythonProject\EndProjekt\final_data\test_data_synthetic"
target_folder = r"C:\Users\MrKoiKoi\PycharmProjects\PythonProject\EndProjekt\final_data\test_data_real"

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
loader = FileLoader(target_folder, pic_target_resolution).load()

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

for label_id in range(len(label_names)):
    indices = np.where(y_class == label_id)[0]
    num_imgs = len(indices)
    cols = 5
    rows = int(np.ceil(num_imgs / cols))

    plt.figure(figsize=(18, 4 * rows))

    for plot_idx, i in enumerate(indices):
        ax = plt.subplot(rows, cols, plot_idx + 1)
        ax.imshow(images[i])

        # Vorhersage (grün)
        xc, yc, w, h = pred_bbox[i] * pic_target_resolution
        x, y = xc - w / 2, yc - h / 2
        ax.add_patch(patches.Rectangle(
            (x, y), w, h, linewidth=2,
            edgecolor='lime', facecolor='none', linestyle='-'
        ))

        # Ground Truth (rot)
        xc, yc, w, h = y_bbox[i] * pic_target_resolution
        x, y = xc - w / 2, yc - h / 2
        ax.add_patch(patches.Rectangle(
            (x, y), w, h, linewidth=2,
            edgecolor='red', facecolor='none', linestyle='--'
        ))

        # Titel
        mark = "✓" if pred_labels[i] == y_class[i] else "✗"
        ax.set_title(
            f"{mark} {label_names[pred_labels[i]]}",
            fontsize=16,
            color='green' if pred_labels[i] == y_class[i] else 'red'
        )
        ax.axis('off')

    correct = np.sum(pred_labels[indices] == y_class[indices])
    plt.suptitle(
        f"{label_names[label_id]} — {correct}/{num_imgs} richtig | Grün=Pred, Rot=Best",
        fontsize=16
    )
    plt.tight_layout()
    plt.show()