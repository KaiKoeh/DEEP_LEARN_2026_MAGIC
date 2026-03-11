import os
import numpy as np
from PIL import Image
from tensorflow import keras
from sklearn.model_selection import train_test_split
from helper_classes.file_loader_class import FileLoader
from helper_classes.config_loader import ConfigLoader
from datetime import datetime


### CONFIG-LOADER
main_folder = os.path.dirname(os.path.abspath(__file__)) + "/"
config_loader = ConfigLoader(main_folder + "config_file.txt")

#### TRAIN_FOLDER
train_folder = config_loader.train_data_real_path

##### TARGET MODEL FOLDER:
output_target_name = "output_" + datetime.now().strftime("%Y_%m_%d_%H_%M")
model_output_folder = config_loader.model_output_path + output_target_name + "/"

#### INIT LOADER
loader = FileLoader(train_folder, config_loader.label_file_path, config_loader.width, config_loader.height).load()

#### FILE-DATA
images = loader.images
y_bbox = loader.y_bbox
y_class = loader.y_class
label_names = loader.label_names

label_amount = len(label_names)
image_amount = len(images)

TEST_SIZE = 0.2
BATCH_SIZE = 25
EPOCHS = 3

print(f"Label-Amount: {label_amount}  Datei-Amount: {image_amount}")
print(f"Train Info train_folder: {train_folder} - TEST_SIZE: {TEST_SIZE}  BATCH_SIZE: {BATCH_SIZE}  EPOCHS: {EPOCHS}")

#### SET-RANDOM SEED
np.random.seed(42)

#### DRAW
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.figure(figsize=(12, 8))

# Welche Klassen sind in den Daten vorhanden?
valid_classes = np.unique(y_class)
view_amount = min(12, len(valid_classes))

for i in range(view_amount):
    klassen_id = valid_classes[i]

    # Erstes Bild dieser Klasse finden
    alle_indices = np.where(y_class == klassen_id)[0]
    erstes_bild_idx = alle_indices[0]

    ax = plt.subplot(3, 4, i + 1)
    ax.imshow(images[erstes_bild_idx])

    # YOLO normalisiert → Pixel
    xc = y_bbox[erstes_bild_idx][0] * config_loader.width
    yc = y_bbox[erstes_bild_idx][1] * config_loader.height
    w  = y_bbox[erstes_bild_idx][2] * config_loader.width
    h  = y_bbox[erstes_bild_idx][3] * config_loader.height

    rect = patches.Rectangle(
        (xc - w / 2, yc - h / 2), w, h,
        linewidth=2, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)
    ax.set_title(label_names[klassen_id], fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.show()

### Train Test-Split
X_train, X_test, y_train_bbox, y_test_bbox, y_train_class, y_test_class  = train_test_split(images, y_bbox, y_class, test_size=TEST_SIZE)

base_model = keras.applications.MobileNetV2(
    input_shape=(config_loader.height, config_loader.width, 3),
    alpha=1.0,
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax"
)

### WICHTIG
base_model.trainable = False

def my_preprocess(x, training):
    return keras.applications.mobilenet_v2.preprocess_input(x)

## MODEL INIT
seq_model = keras.models.Sequential()

base_input = keras.layers.Input((config_loader.height, config_loader.width, 3))
seq_model.add(base_input)

#### RANDOM TRAIN DATA MANIPULATION
seq_model.add(keras.layers.RandomBrightness(0.2, value_range=(0, 255)))
seq_model.add(keras.layers.RandomContrast(0.2, value_range=(0, 255)))
seq_model.add(keras.layers.Lambda(my_preprocess))

##model.add(keras.layers.RandomRotation(0.02, fill_mode="nearest", interpolation="bilinear", seed=None, fill_value=128))
##model.add(keras.layers.RandomZoom([-0.2, 0.0], fill_value=180))
##model.add(keras.layers.RandomFlip(mode="horizontal", seed=0))
##model.add(keras.layers.RandomContrast(0.2, value_range=(0, 255), seed=None))
##model.add(keras.layers.RandomBrightness([-0.3, 0.3], value_range=(0, 255), seed=0))
##model.add(keras.layers.RandomGaussianBlur(factor=1.0, kernel_size=3, sigma=1.0, value_range=(0, 255), data_format=None, seed=None))


### ADD BASE MODEL
seq_model.add(base_model)

# Output > Teilung auf Klasse und BBox
shared = seq_model.outputs[0]


# Verzweigung für zwei Outputs: Klasse && BBox

# --- Dense Schichten für die Klasse
class_x = keras.layers.GlobalAveragePooling2D()(shared)
class_x = keras.layers.Dropout(0.3)(class_x)
class_output = keras.layers.Dense(label_amount, activation="softmax", name="class")(class_x)


# --- Dense Schichten für die BBox
bbox_x = keras.layers.Conv2D(64, (1, 1), activation='relu')(shared)
bbox_x = keras.layers.Conv2D(16, (3, 3), activation='relu')(bbox_x)
bbox_x = keras.layers.Flatten()(bbox_x)
bbox_x = keras.layers.Dropout(0.3)(bbox_x)
bbox_x = keras.layers.Dense(64, activation='relu')(bbox_x)
bbox_x = keras.layers.Dropout(0.2)(bbox_x)

# --- sigmoid 0-1 kanten koordinaten in Prozent im Bild
bbox_output = keras.layers.Dense(4, activation='sigmoid', name="bbox")(bbox_x)

model = keras.Model(inputs=base_input, outputs=[class_output, bbox_output])

model.summary()

#### KERAS OPTIMIZER
optimizer = keras.optimizers.Adam(learning_rate=0.001)  # 0.002

model.compile(loss=["sparse_categorical_crossentropy", "mse"], loss_weights=[0.2, 0.8], optimizer=optimizer, metrics={"class": "accuracy", "bbox": "r2_score"})


#### EARLY STOPPING
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

history = model.fit(X_train, [y_train_class, y_train_bbox], batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, verbose=1, callbacks=[early_stopping])

####### Nach dem Training:
# Output-Verzeichnis erstellen + Label-Datei kopieren
import shutil
os.makedirs(model_output_folder, exist_ok=True)
shutil.copy2(config_loader.label_file_path, model_output_folder + "label_file.txt")


#### MODEL SPEICHERN

model.save(model_output_folder + "model.keras")
model.save_weights(model_output_folder + "model.weights.h5")

##### AUSWERTUNG

# Lernkurven - BBox MAE
plt.plot(history.history['bbox_r2_score'], label='Train')
plt.plot(history.history['val_bbox_r2_score'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('BBox R2 Score')
plt.legend()
plt.show()
plt.savefig(model_output_folder + "bbox_r2_score.png")

# Lernkurven - Class Accuracy
plt.plot(history.history['class_accuracy'], label='Train')
plt.plot(history.history['val_class_accuracy'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Class Accuracy')
plt.legend()
plt.show()
plt.savefig(model_output_folder + "class_accuracy.png")

# Lernkurven - Loss
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig(model_output_folder + "loss.png")

# Loss
actual_epochs = len(history.history['loss'])
x_train_amount = len(X_train)
x_test_amount = len(X_test)
relative_path = train_folder.replace(main_folder, "")

last_bbox_r2 = round(history.history['val_bbox_r2_score'][-1], 3)
last_class_acc = round(history.history['val_class_accuracy'][-1], 3)
last_loss = round(history.history['val_loss'][-1], 3)

### LOG & INFO
logPart_1 = f"Train Info train_folder: {relative_path}"
logPart_2 = f"TEST_SIZE: {TEST_SIZE}  BATCH_SIZE: {BATCH_SIZE}  EPOCHS: {EPOCHS}"
logPart_3 = f"Epochen durchgelaufen: {actual_epochs}"
logPart_4 = f"Klassen-Anzahl: {label_amount}  Daten-Anzahl: {image_amount} > Durchschnitt Bilder per Label {round(image_amount / label_amount,1)} "
logPart_5 = f"Train-Test Split-Size {TEST_SIZE} Train Daten-Anzahl: {x_train_amount} Test Daten-Anzahl: {x_test_amount}"
logPart_6 = f"Image-Size width: {config_loader.width}  height: {config_loader.height}"
logPart_7 = F"Letzte Messung BBBox-R2: {last_bbox_r2} Class-accuracy: {last_class_acc}  Val-loss: {last_loss}"


print(logPart_1)
print(logPart_2)
print(logPart_3)
print(logPart_4)
print(logPart_5)
print(logPart_6)
print(logPart_7)
print("Labels:")
print(label_names)

log_path = os.path.join(model_output_folder, "model_log.txt")
with open(log_path, "w") as f:
    f.write("Model Compile Infos:")
    f.write(logPart_1 + "\n")
    f.write(logPart_2 + "\n")
    f.write(logPart_3 + "\n")
    f.write(logPart_4 + "\n")
    f.write(logPart_5 + "\n")
    f.write(logPart_6 + "\n")
    f.write(logPart_7 + "\n")
    f.write("Labels:\n")
    f.write(str(label_names) + "\n")



