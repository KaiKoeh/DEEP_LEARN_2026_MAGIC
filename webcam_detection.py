import os
import cv2
import numpy as np
from tensorflow import keras
from helper_classes.config_loader import ConfigLoader

### CONFIG-LOADER
main_folder = os.path.dirname(os.path.abspath(__file__)) + "/"
config = ConfigLoader(main_folder + "config_file.txt")

#### MODEL FOLDER
output_folder = "output_synthetic_big"

#### MODEL & LABELS LADEN
model_path = config.model_output_path + output_folder + "/model.keras"
label_path = config.model_output_path + output_folder + "/label_file.txt"

EXPORT_W = config.width    # 192
EXPORT_H = config.height   # 256

### CONFIDENCE SCHWELLENWERT
MIN_CONFIDENCE = 0.90

def my_preprocess(x, training):
    return keras.applications.mobilenet_v2.preprocess_input(x)

print(f"Lade Model: {model_path}")
model = keras.models.load_model(
    model_path,
    custom_objects={"my_preprocess": my_preprocess}
)

## Camera Check
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Kamera {i}: verfügbar")
        cap.release()
    else:
        print(f"Kamera {i}: nicht gefunden")

### Labels laden
label_names = {}
with open(label_path) as f:
    for i, line in enumerate(f.readlines()):
        label_names[i] = line.strip()

print(f"Model geladen: {len(label_names)} Klassen")

CARD_OVERLAY_HEIGHT = 600  # Höhe des Preview Bildes in Pixel
scryfall_folder = config.scryfall_cards_path
card_images = {}

for i, name in label_names.items():
    img_path = os.path.join(scryfall_folder, name + ".png")
    if os.path.exists(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # Auf Overlay-Höhe skalieren (Ratio beibehalten)
        scale = CARD_OVERLAY_HEIGHT / img.shape[0]
        overlay_w = int(img.shape[1] * scale)
        img = cv2.resize(img, (overlay_w, CARD_OVERLAY_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
        card_images[i] = img

print(f"Karten-Bilder geladen: {len(card_images)}/{len(label_names)}")


cap = cv2.VideoCapture(1)

if not cap.isOpened():
    exit()


label_text = ""
text_size = None
overlay = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ### Frame Höhe und Breite
    frame_h, frame_w = frame.shape[:2]
    target_ratio = EXPORT_W / EXPORT_H  # 0.75

    if frame_w / frame_h > target_ratio:
        # Frame ist zu breit → links/rechts abschneiden
        crop_w = int(frame_h * target_ratio)
        crop_h = frame_h
    else:
        # Frame ist zu hoch → oben/unten abschneiden
        crop_w = frame_w
        crop_h = int(frame_w / target_ratio)

    crop_x = (frame_w - crop_w) // 2
    crop_y = (frame_h - crop_h) // 2
    cropped = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

    resized = cv2.resize(cropped, (EXPORT_W, EXPORT_H))

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    ### Prediction
    input_img = np.expand_dims(rgb, axis=0).astype(np.float32)
    pred_class, pred_bbox = model.predict(input_img, verbose=0)

    ### Ergebnis auswerten
    class_id = np.argmax(pred_class[0])
    confidence = pred_class[0][class_id]
    card_name = label_names.get(class_id, f"ID {class_id}")

    ### BBox
    xc = pred_bbox[0][0] * crop_w
    yc = pred_bbox[0][1] * crop_h
    bw = pred_bbox[0][2] * crop_w
    bh = pred_bbox[0][3] * crop_h

    x1 = int(xc - bw / 2) + crop_x
    y1 = int(yc - bh / 2) + crop_y
    x2 = int(xc + bw / 2) + crop_x
    y2 = int(yc + bh / 2) + crop_y

    if confidence >= MIN_CONFIDENCE:
        label_text = f"{confidence * 100:.1f}% {card_name}"
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        if class_id in card_images:
            overlay = card_images[class_id]

    if text_size is not None:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), (0, 255, 0), -1)
        cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    if overlay is not None:
        oh, ow = overlay.shape[:2]
        pad = 10  # Abstand vom Rand

        # Position oben rechts
        ox = frame_w - ow - pad
        oy = pad

        # Alpha-Blending
        alpha = overlay[:, :, 3] / 255.0
        for c in range(3):
            frame[oy:oy+oh, ox:ox+ow, c] = (
                overlay[:, :, c] * alpha + frame[oy:oy+oh, ox:ox+ow, c] * (1 - alpha)
            ).astype(np.uint8)

    ### Crop-Bereich anzeigen (gestrichelte Linien)
    cv2.line(frame, (crop_x, 0), (crop_x, frame_h), (100, 100, 255), 1)
    cv2.line(frame, (crop_x + crop_w, 0), (crop_x + crop_w, frame_h), (100, 100, 255), 1)

    cv2.imshow("Card Detection", frame)

    ### Beenden mit 'q' >>>> Webcam könnte sonst hängen!!!!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

### sollte
cap.release()
cv2.destroyAllWindows()

