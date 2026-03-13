import os
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import confusion_matrix
from helper_classes.file_loader_class import FileLoader
from helper_classes.config_loader import ConfigLoader

### SET RANDOM SEED
np.random.seed(42)

#### MODEL FOLDER
output_folder = "output_synthetic_small" ## "output_2026_03_08_20_20" ## output_2026_03_07_23_43

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
label_path = config.model_output_path + output_folder + "/label_file.txt"
print("label_path", label_path)
loader = FileLoader(target_folder, label_path, EXPORT_W, EXPORT_H).load()

#### FILE-DATA
images = loader.images
y_bbox = loader.y_bbox
y_class = loader.y_class
label_names = loader.label_names


print(f"{len(images)} Testbilder | Labels: {label_names}")




# --- Evaluate ---
results = model.evaluate(images, [y_class, y_bbox])

test_loss = round(results[0], 3)
bbox_accuracy = round(results[3], 3)
class_accuracy = round(results[4], 3)

print(f"Test Loss:       {test_loss}")
print(f"BBox R2 Score:  {bbox_accuracy}")
print(f"Class Accuracy:   {class_accuracy}")


# --- Predict ---
pred_class, pred_bbox = model.predict(images)
pred_labels = np.argmax(pred_class, axis=1)

# --- Intersection over Union

def calculate_iou(pred_bbox, true_bbox):

    # YOLO: [xc, yc, w, h] normalisiert 0-1
    px, py, pw, ph = pred_bbox
    tx, ty, tw, th = true_bbox

    # In Ecken umrechnen (x1, y1, x2, y2)
    pred_x1 = px - pw / 2
    pred_y1 = py - ph / 2
    pred_x2 = px + pw / 2
    pred_y2 = py + ph / 2

    true_x1 = tx - tw / 2
    true_y1 = ty - th / 2
    true_x2 = tx + tw / 2
    true_y2 = ty + th / 2

    # Überlappung (Intersection)
    inter_x1 = max(pred_x1, true_x1)
    inter_y1 = max(pred_y1, true_y1)
    inter_x2 = min(pred_x2, true_x2)
    inter_y2 = min(pred_y2, true_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Union
    pred_area = pw * ph
    true_area = tw * th
    union_area = pred_area + true_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


ious = []
for i in range(len(y_bbox)):
    iou = calculate_iou(pred_bbox[i], y_bbox[i])
    ious.append(iou)

avg_iou = float(np.mean(ious))
avg_iou = round(avg_iou,2)

print(f"Durchschnitt IoU: {avg_iou}")


# --- Visualisierung ---

PLOT_COLS = 5
PLOT_ROWS = 5
PLOT_COUNT = PLOT_COLS * PLOT_ROWS

correct_indices = np.where(pred_labels == y_class)[0]
wrong_indices = np.where(pred_labels != y_class)[0]

np.random.shuffle(correct_indices)
np.random.shuffle(wrong_indices)

total_amount = len(y_class)
correct_amount = len(correct_indices)
wrong_amount = len(wrong_indices)
correct_percent = np.round(correct_amount / total_amount * 100, 1)

print(f"Klassifikation: {correct_amount}/{total_amount} richtig ({correct_percent}%)")
print(f"folder: {target_folder}")

MAX_LABEL_LEN = 13

table_figure = [
    (f"RICHTIG — {correct_amount}/{total_amount}", correct_indices[:PLOT_COUNT], 'green'),
    (f"FALSCH — {wrong_amount}/{total_amount}", wrong_indices[:PLOT_COUNT], 'red'),
]


for title, indices, color in table_figure:

    if len(indices) == 0:
        continue

    rows = int(np.ceil(len(indices) / PLOT_COLS))
    fig, axes = plt.subplots(rows, PLOT_COLS, figsize=(15, 4 * rows))

    # axes immer 2D
    if rows == 1:
        axes = [axes]

    # Alle Felder ausblenden
    for r in range(rows):
        for c in range(PLOT_COLS):
            axes[r][c].axis('off')

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


        pred_name = label_names[pred_labels[i]]
        pred_name = pred_name[:MAX_LABEL_LEN]
        true_name = label_names[y_class[i]]
        true_name = true_name[:MAX_LABEL_LEN]

        pred_conf = pred_class[i][pred_labels[i]] * 100
        true_conf = pred_class[i][y_class[i]] * 100


        if pred_labels[i] == y_class[i]:
            ax.set_title(f"✓ {pred_name} ({pred_conf:.1f}%)", fontsize=14, color='green')
        else:
            ax.set_title(f"✗ {pred_name} ({pred_conf:.1f}%)\n→ {true_name} ({true_conf:.1f}%)", fontsize=14, color='red')

    plt.suptitle(title, fontsize=14, color=color, y=0.99)
    plt.tight_layout()
    plt.show()


# --- Confusion Matrix ---

# Nur Klassen die in Testdaten vorkommen
active_classes = sorted(set(y_class.tolist()))

font_size_numbers = 26
font_size_names = 22
font_size_title = 28


active_names = []
for c in active_classes:
    name = label_names[c]
    name = name[:MAX_LABEL_LEN]   # auf max Länge kürzen
    active_names.append(name)

cm = confusion_matrix(y_class, pred_labels, labels=active_classes)

# Größe dynamisch: min 15, max 40
fig_size = max(15, min(40, len(active_classes) * 0.5))

fig, ax = plt.subplots(figsize=(fig_size, fig_size))
im = ax.imshow(cm, cmap='Blues', interpolation='nearest')

title = f'Confusion Matrix: {correct_amount}/{total_amount} ({correct_percent}%) - BBox R2:{bbox_accuracy} IoU:{avg_iou}'

# Zahlen in die Zellen
for i in range(len(active_classes)):
    for j in range(len(active_classes)):
        val = cm[i, j]
        if val > 0:
            color = 'white' if val > cm.max() / 2 else 'black'
            ax.text(j, i, str(val), ha='center', va='center', fontsize=28, color=color)

ax.set_xticks(range(len(active_names)))
ax.set_yticks(range(len(active_names)))
ax.set_xticklabels(active_names, rotation=90, fontsize=font_size_names)
ax.set_yticklabels(active_names, fontsize=font_size_names)

##ax.set_xlabel('Predicted', fontsize=16)
##ax.set_ylabel('Daten', fontsize=16)
ax.set_title(title, fontsize=font_size_title)

##plt.colorbar(im, ax=ax, label='Anzahl Predictions', shrink=0.8)
plt.tight_layout()
plt.show()

#####  BArs für große Datensätze

MIN_LABEL_COUNT = 10

bar_labels = []
bar_correct = []
bar_total = []

for cls_id in sorted(set(y_class.tolist())):
    cls_indices = np.where(y_class == cls_id)[0]
    cls_total = len(cls_indices)

    if cls_total < MIN_LABEL_COUNT:
        continue

    cls_correct = np.sum(pred_labels[cls_indices] == y_class[cls_indices])

    bar_labels.append(label_names[cls_id][:MAX_LABEL_LEN])
    bar_correct.append(cls_correct)
    bar_total.append(cls_total)

bar_correct = np.array(bar_correct)
bar_total = np.array(bar_total)
bar_wrong = bar_total - bar_correct

# Plot
fig_height = max(8, len(bar_labels) * 0.4)
fig, ax = plt.subplots(figsize=(12, fig_height))

y_pos = np.arange(len(bar_labels))

ax.barh(y_pos, bar_correct, color='green', label='Richtig')
ax.barh(y_pos, bar_wrong, left=bar_correct, color='red', label='Falsch')

for i in range(len(bar_labels)):
    ax.text(bar_total[i] / 2, y_pos[i], f"{bar_correct[i]}/{bar_total[i]}",
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')

ax.set_yticks(y_pos)
ax.set_yticklabels(bar_labels, fontsize=12)
ax.set_title(f'Accuracy pro Klasse (min. {MIN_LABEL_COUNT} Bilder) — {correct_amount}/{total_amount} ({correct_percent}%) IoU: {avg_iou}', fontsize=14)
ax.legend(loc='lower right')
ax.invert_yaxis()

plt.tight_layout()
plt.show()