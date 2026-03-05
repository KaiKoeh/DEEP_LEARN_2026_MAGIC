import os
import numpy as np
from PIL import Image
from tensorflow import keras
from sklearn.model_selection import train_test_split
from file_loader_class import FileLoader
from config_loader import ConfigLoader


### PC
##main_folder = r"C:\Users\MrKoiKoi\PycharmProjects\PythonProject\EndProjekt" + "\\"

### MAC
main_folder = r"/Users/kaikohrsen/Documents/schulung/PythonWeekly/deep_learn_project/"


target_folder = main_folder + "/final_data/train_data_synthetic"
save_file = "mtg_detector_synthetic"


### CONFIG-LOADER
config_loader = ConfigLoader(main_folder+"config_file.txt")

#### INIT LOADER
loader = FileLoader(target_folder, config_loader.width, config_loader.height).load()

#### FILE-DATA
images = loader.images
y_bbox = loader.y_bbox
y_class = loader.y_class
label_names = loader.label_names

label_amount = len(label_names)

print("labels")
print(label_names)
#### DRAW

import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.figure(figsize=(12, 8))

for i in range(label_amount):
    ax = plt.subplot(2, 3, i + 1)
    idx = np.where(y_class == i)[0][0]
    ax.imshow(images[idx])

    # YOLO: normalisiert → Pixel
    xc = y_bbox[idx][0] * config_loader.width
    yc = y_bbox[idx][1] * config_loader.height
    w  = y_bbox[idx][2] * config_loader.width
    h  = y_bbox[idx][3] * config_loader.height
    x = xc - w / 2
    y = yc - h / 2

    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    ax.set_title(label_names[i], fontsize=10)
    ax.axis('off')


plt.tight_layout()
plt.show()


X_train, X_test, y_train_bbox, y_test_bbox, y_train_class, y_test_class  = train_test_split(images, y_bbox, y_class, test_size=0.2, random_state=240)

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

seq_model.add(keras.layers.RandomBrightness(0.2, value_range=(0, 255)))
seq_model.add(keras.layers.RandomContrast(0.2, value_range=(0, 255)))
seq_model.add(keras.layers.Lambda(my_preprocess))

### ADD BASE MODEL
seq_model.add(base_model)

# Output > Teilung auf Klasse und BBox
shared = seq_model.outputs[0]


# Verzweigung für zwei Outputs: Klasse && BBox

# --- Dense Schichten für die Klasse
class_x = keras.layers.GlobalAveragePooling2D()(shared)
class_x = keras.layers.Dropout(0.3)(class_x)
class_output = keras.layers.Dense(len(label_names), activation="softmax", name="class")(class_x)


# --- Dense Schichten für die BBox
bbox_x = keras.layers.Conv2D(64, (1, 1), activation='relu')(shared)   # (8,8,1024) → (8,8,64)
bbox_x = keras.layers.Conv2D(16, (3, 3), activation='relu')(bbox_x)   # (8,8,64)  → (6,6,16)
bbox_x = keras.layers.Flatten()(bbox_x)                                # → 576
bbox_x = keras.layers.Dropout(0.3)(bbox_x)
bbox_x = keras.layers.Dense(64, activation='relu')(bbox_x)             # → 64
bbox_x = keras.layers.Dropout(0.2)(bbox_x)

# --- sigmoid 0-1 kanten koordinaten in Prozent im Bild
bbox_output = keras.layers.Dense(4, activation='sigmoid', name="bbox")(bbox_x)

model = keras.Model(inputs=base_input, outputs=[class_output, bbox_output])

model.summary()

#### KERAS OPTIMIZER
optimizer = keras.optimizers.Adam(learning_rate=0.001)  # 0.002

model.compile(loss=["sparse_categorical_crossentropy", "mse"], loss_weights=[0.2, 0.8], optimizer=optimizer, metrics={"class": "accuracy", "bbox": "r2_score"})

#### Brauche -- Intersection over Union, IOU

history = model.fit(X_train, [y_train_class, y_train_bbox], batch_size=100, epochs=150, validation_split=0.2, verbose=1)

# Nach dem Training:
model.save(save_file + ".keras")
model.save_weights(save_file + ".weights.h5")


##### AUSWERTUNG
# Lernkurven - BBox MAE
plt.plot(history.history['bbox_r2_score'], label='Train')
plt.plot(history.history['val_bbox_r2_score'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('BBox R2 Score')
plt.legend()
plt.show()

# Lernkurven - Class Accuracy
plt.plot(history.history['class_accuracy'], label='Train')
plt.plot(history.history['val_class_accuracy'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Class Accuracy')
plt.legend()
plt.show()

# Lernkurven - Loss
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()



print(label_names)

