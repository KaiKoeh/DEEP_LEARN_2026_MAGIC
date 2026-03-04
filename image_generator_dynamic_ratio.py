import numpy
from PIL import Image
import os
import numpy as np
from PIL import Image
import random
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cv2 ### >>> opencv
import shutil


### PROJECT FOLDER
main_folder = "/Users/kaikohrsen/Documents/schulung/PythonWeekly/deep_learn_project/"
##main_folder = r"C:\Users\MrKoiKoi\PycharmProjects\PythonProject\EndProjekt" + "\\"

####### FILE-LOAD ########
bg_folder = main_folder + "image_generator/backgrounds"
card_folder = main_folder + "image_generator/cards"
label_file = main_folder + "label_file.txt"


####### EXPORT ########
output_folder = main_folder + "final_data"
train_folder = output_folder + "/train_data_synthetic"
test_folder = output_folder + "/test_data_synthetic"

train_split_value = 0.8

### VARS
label_names = {}
backgrounds = []

## CARDS
cards = []
card_classes = []

### Erzeugte Backgrounds
bg_canvases = []


#### Erzeugungs Varianten
BACKGROUND_VARIATIONS = 1 ## 20
CARDS_PER_CANVAS = 1 ## 20

######## OUTPUT RATIO & SIZE ########
OUTPUT_SIZE = 256                    # Längste Seite in Pixel
OUTPUT_RATIO = (3, 4)                # Breite:Höhe (3:4 = Handy Hochkant)

######## BACKGROUND SETUP
BG_CANVAS_LONG = 1024               # Längste Seite des Canvas
BG_SHIFT_RANGE = 150                # max Pixel Verschiebung
BG_ROTATE_RANGE = 90                # max Grad Rotation in beide Richtungen
BG_ZOOM_RANGE = (1.3, 3.0)

######## KARTE HINZUFÜGEN ----
CARD_SCALE_RANGE = (0.35, 0.7)      # Karte nimmt xx-xx% des Canvas ein (bezogen auf Höhe)
CARD_ROTATE_RANGE = 35              # PROZENT Rotation
CARD_PADDING = 0.15                 # xx% Abstand vom Rand der Karten position, ACHTUNG KIPPEN UND PAD Schnitt testen!

######## KIPPEN DER PERSPEKTIVE + CUT OUT
PERSPECTIVE_SHIFT = 0.125            # MAX KIPP WINKEL DER PERSPEKTIVE
PERSPECTIVE_PAD = 0.2                # extra Rand zum Wegschneiden, da Schwarze Ränder beim Kippen

######## COLOR EFFEKT EBENE

## ÜBER DAS BILD
NOISE_RANDOM = (0.02, 0.1)          # NOISE EFFEKT RANDOM INTENSIVITÄT
MOTION_BLUR = (3, 10)               # VERWACKELUNGS EFFEKT RANDOM INTENSIVITÄT
OVERALL_BRIGHTNESS = (0.8, 1.2)     # HELLIGKEIT > ÜBER DAS GESAMTE BILD
OVERALL_CONTRAST = (0.8, 1.2)       # KONTRAST > ÜBER DAS GESAMTE BILD
OVERALL_SATURATION = (0.8, 1.2)     # SÄTTIGUNG > ÜBER DAS GESAMTE BILD

## EFFEKTE separiert > Background und Card
ENTITY_BRIGHTNESS = (0.8, 1.2)      ### HELLIGKEIT > BG / CARD SEPARAT
ENTITY_CONTRAST = (0.8, 1.2)        ### KONTRAST > BG / CARD SEPARAT
ENTITY_SATURATION = (0.8, 1.2)      ### SÄTTIGUNG > BG / CARD SEPARAT

### EXPORT DATA
DELETE_OLD_EXPORT = True

def add_camera_noise(image, intensity=0.10):
    noise = np.random.normal(0, intensity * 255, image.shape)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def add_motion_blur(image, max_kernel=7):
    max_kernel = max(3, max_kernel)
    size = random.choice(range(3, max_kernel + 1, 2))
    angle = random.uniform(0, 360)

    # Motion-Blur Kernel erstellen
    kernel = np.zeros((size, size))
    kernel[size // 2, :] = 1.0 / size

    # Kernel rotieren
    center = (size // 2, size // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (size, size))
    kernel = kernel / kernel.sum()

    return cv2.filter2D(image, -1, kernel)

def augment_color(image, entity=False):
    img = Image.fromarray(image)

    from PIL import ImageEnhance

    if entity:
        brightness = random.uniform(*ENTITY_BRIGHTNESS)
    else:
        brightness = random.uniform(*OVERALL_BRIGHTNESS)

    img = ImageEnhance.Brightness(img).enhance(brightness)

    if entity:
        contrast = random.uniform(*ENTITY_CONTRAST)
    else:
        contrast = random.uniform(*OVERALL_CONTRAST)

    img = ImageEnhance.Contrast(img).enhance(contrast)

    if entity:
        saturation = random.uniform(*ENTITY_SATURATION)
    else:
        saturation = random.uniform(*OVERALL_SATURATION)

    img = ImageEnhance.Color(img).enhance(saturation)

    return np.array(img)

### BERECHNUNG RATIO SIZE & CANVAS

if OUTPUT_RATIO[0] >= OUTPUT_RATIO[1]:
    EXPORT_W = OUTPUT_SIZE
    EXPORT_H = int(OUTPUT_SIZE * OUTPUT_RATIO[1] / OUTPUT_RATIO[0])
else:
    EXPORT_H = OUTPUT_SIZE
    EXPORT_W = int(OUTPUT_SIZE * OUTPUT_RATIO[0] / OUTPUT_RATIO[1])


if OUTPUT_RATIO[0] >= OUTPUT_RATIO[1]:
    BG_CANVAS_W = BG_CANVAS_LONG
    BG_CANVAS_H = int(BG_CANVAS_LONG * OUTPUT_RATIO[1] / OUTPUT_RATIO[0])
else:
    BG_CANVAS_H = BG_CANVAS_LONG
    BG_CANVAS_W = int(BG_CANVAS_LONG * OUTPUT_RATIO[0] / OUTPUT_RATIO[1])

print(f"Canvas-Größe: {BG_CANVAS_W}x{BG_CANVAS_H}")

print(f"Export-Größe: {EXPORT_W}x{EXPORT_H} (Ratio {OUTPUT_RATIO[0]}:{OUTPUT_RATIO[1]})")


#### Label_Names aus der Datei laden
with open(label_file) as f:
    for i, name in enumerate(f.readlines()):
        label_names[i] = name.strip()


for filename in sorted(os.listdir(card_folder)):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        card_name = filename.split(".")[0]

        ### Erst prüfen ob Label existiert, dann laden
        for class_id, label in label_names.items():
            if label == card_name:
                img = Image.open(os.path.join(card_folder, filename))
                cards.append(np.array(img))
                card_classes.append(class_id)
                print(f"  {filename} → Klasse {class_id} ({label})")
                break

for filename in sorted(os.listdir(bg_folder)):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        img = Image.open(os.path.join(bg_folder, filename))
        backgrounds.append(np.array(img))
        print(f"  {filename} → {img.size}")




#### Background Prepair to CANVAS

for bg in backgrounds:
    h, w = bg.shape[:2]

    # Sicherstellen, dass BG groß genug für Canvas ist
    if w < BG_CANVAS_W or h < BG_CANVAS_H:
        scale = max(BG_CANVAS_W / w, BG_CANVAS_H / h)
        img = Image.fromarray(bg).resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        bg = np.array(img)
        h, w = bg.shape[:2]

    cx = (w - BG_CANVAS_W) // 2
    cy = (h - BG_CANVAS_H) // 2
    canvas = bg[cy:cy + BG_CANVAS_H, cx:cx + BG_CANVAS_W].copy()
    bg_canvases.append(canvas)

    for v in range(BACKGROUND_VARIATIONS):
        shift_x = random.randint(-BG_SHIFT_RANGE, BG_SHIFT_RANGE)
        shift_y = random.randint(-BG_SHIFT_RANGE, BG_SHIFT_RANGE)

        left = max(0, min(cx + shift_x, w - BG_CANVAS_W))
        top = max(0, min(cy + shift_y, h - BG_CANVAS_H))

        crop = bg[top:top + BG_CANVAS_H, left:left + BG_CANVAS_W].copy()
        crop = augment_color(crop, entity=True)

        # Rotieren (quadratisch zoomen damit nach Rotation genug Fläche bleibt)
        angle = random.uniform(-BG_ROTATE_RANGE, BG_ROTATE_RANGE)
        crop_img = Image.fromarray(crop).rotate(angle, resample=Image.BICUBIC, fillcolor=(0, 0, 0))

        # Random Zoom: hochskalieren, dann Canvas-Größe ausschneiden
        zoom = random.uniform(*BG_ZOOM_RANGE)
        zoomed_w = int(BG_CANVAS_W * zoom)
        zoomed_h = int(BG_CANVAS_H * zoom)
        crop_img = crop_img.resize((zoomed_w, zoomed_h), Image.LANCZOS)

        zl = (zoomed_w - BG_CANVAS_W) // 2 + random.randint(-30, 30)
        zt = (zoomed_h - BG_CANVAS_H) // 2 + random.randint(-30, 30)
        zl = max(0, min(zl, zoomed_w - BG_CANVAS_W))
        zt = max(0, min(zt, zoomed_h - BG_CANVAS_H))

        crop_img = crop_img.crop((zl, zt, zl + BG_CANVAS_W, zt + BG_CANVAS_H))
        bg_canvases.append(np.array(crop_img))


############ GENERATE IMGAGES

generated_images = []
generated_labels_bbox = []   # (xc, yc, w, h) normalisiert
generated_labels_class = []  # class_id

for canvas in bg_canvases:
    for _ in range(CARDS_PER_CANVAS):
        card_idx = random.randint(0, len(cards) - 1)
        card = Image.fromarray(augment_color(cards[card_idx], entity=True))

        scale = random.uniform(*CARD_SCALE_RANGE)
        card_h = int(BG_CANVAS_H * scale)
        card_w = int(card_h * (card.width / card.height))
        card_resized = card.resize((card_w, card_h), Image.LANCZOS)

        angle = random.uniform(-CARD_ROTATE_RANGE, CARD_ROTATE_RANGE)
        card_rotated = card_resized.rotate(angle, expand=True, resample=Image.BICUBIC)
        rot_w, rot_h = card_rotated.size

        pad_x = int(BG_CANVAS_W * CARD_PADDING)
        pad_y = int(BG_CANVAS_H * CARD_PADDING)
        max_x = BG_CANVAS_W - rot_w - pad_x
        max_y = BG_CANVAS_H - rot_h - pad_y

        if max_x < pad_x or max_y < pad_y:
            continue

        pos_x = random.randint(pad_x, max_x)
        pos_y = random.randint(pad_y, max_y)
        result = Image.fromarray(canvas.copy())
        result.paste(card_rotated, (pos_x, pos_y), card_rotated.convert("RGBA").split()[3])

        xc = (pos_x + rot_w / 2) / BG_CANVAS_W
        yc = (pos_y + rot_h / 2) / BG_CANVAS_H
        bw = rot_w / BG_CANVAS_W
        bh = rot_h / BG_CANVAS_H

        generated_images.append(np.array(result))
        generated_labels_bbox.append([xc, yc, bw, bh])
        generated_labels_class.append(card_classes[card_idx])  # ← Klasse aus label_file.txt

# In NumPy Arrays konvertieren
generated_images = np.array(generated_images)
generated_labels_bbox = np.array(generated_labels_bbox, dtype=np.float32)
generated_labels_class = np.array(generated_labels_class, dtype=np.int32)


######### KIPPEN DER PERSPEKTIVE + CUT OUT

for i in range(len(generated_images)):
    img = generated_images[i]

    # 1) Bild vergrößern → extra Rand für Perspektive
    pad_px_w = int(BG_CANVAS_W * PERSPECTIVE_PAD)
    pad_px_h = int(BG_CANVAS_H * PERSPECTIVE_PAD)
    big_w = BG_CANVAS_W + 2 * pad_px_w
    big_h = BG_CANVAS_H + 2 * pad_px_h
    big_img = np.array(
        Image.fromarray(img).resize((big_w, big_h), Image.LANCZOS)
    )

    # 2) Perspektive kippen
    shift_w = int(big_w * PERSPECTIVE_SHIFT)
    shift_h = int(big_h * PERSPECTIVE_SHIFT)
    pts1 = np.float32([[0, 0], [big_w, 0], [0, big_h], [big_w, big_h]])
    pts2 = np.float32([
        [random.randint(0, shift_w), random.randint(0, shift_h)],
        [big_w - random.randint(0, shift_w), random.randint(0, shift_h)],
        [random.randint(0, shift_w), big_h - random.randint(0, shift_h)],
        [big_w - random.randint(0, shift_w), big_h - random.randint(0, shift_h)]
    ])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(big_img, matrix, (big_w, big_h))

    # 3) Mitte ausschneiden → schwarze hoffentlich Ränder weg
    generated_images[i] = warped[pad_px_h:pad_px_h + BG_CANVAS_H, pad_px_w:pad_px_w + BG_CANVAS_W]

    # 4) BBox transformieren
    xc, yc, bw, bh = generated_labels_bbox[i]

    # BBox-Koordinaten auf big_size skalieren
    x1 = (xc - bw / 2) * big_w
    y1 = (yc - bh / 2) * big_h
    x2 = (xc + bw / 2) * big_w
    y2 = (yc + bh / 2) * big_h

    corners = np.float32([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
    corners = corners.reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(corners, matrix)
    transformed = transformed.reshape(-1, 2)

    # Auf Crop-Bereich umrechnen (pad_px abziehen)
    new_x1 = max(0, transformed[:, 0].min() - pad_px_w)
    new_y1 = max(0, transformed[:, 1].min() - pad_px_h)
    new_x2 = min(BG_CANVAS_W, transformed[:, 0].max() - pad_px_w)
    new_y2 = min(BG_CANVAS_H, transformed[:, 1].max() - pad_px_h)

    generated_labels_bbox[i] = [
        (new_x1 + new_x2) / 2 / BG_CANVAS_W,
        (new_y1 + new_y2) / 2 / BG_CANVAS_H,
        (new_x2 - new_x1) / BG_CANVAS_W,
        (new_y2 - new_y1) / BG_CANVAS_H
    ]


### GESAMT EFFEKTE ÜBER DAS GANZE BILD
for i in range(len(generated_images)):
    noise_random  = random.uniform(*NOISE_RANDOM)
    generated_images[i] = add_camera_noise(generated_images[i], noise_random)
    blur_random = int(random.uniform(*MOTION_BLUR))
    generated_images[i] = add_motion_blur(generated_images[i], blur_random)
    generated_images[i] = augment_color(generated_images[i], False)


##### SHUFFLE IMAGES

indices = np.arange(len(generated_images))
np.random.shuffle(indices)

generated_images = generated_images[indices]
generated_labels_bbox = generated_labels_bbox[indices]
generated_labels_class = generated_labels_class[indices]


## Alte Daten löschen
if(DELETE_OLD_EXPORT):
    if os.path.exists(train_folder):
        shutil.rmtree(train_folder)

    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)


## Folder erstellen, falls nicht vorhanden
if not os.path.exists(train_folder):
    os.makedirs(train_folder)

if not os.path.exists(test_folder):
    os.makedirs(test_folder)


split_idx = int(len(generated_images) * train_split_value)

### SPEICHERN EINER NUMMER NACH DEM GENMERIERTEN BILD RANDOM
### (damit bestehende mit geringer Wahrscheinlichkeit bilder nicht überschrieben werden)

RANDOM_MAX = 9999

for i in range(len(generated_images)):
    if i < split_idx:
        folder = train_folder
    else:
        folder = test_folder

    cls = generated_labels_class[i]
    name = label_names[cls]

    rand = random.randint(1, RANDOM_MAX)
    digits = len(str(RANDOM_MAX))

    filename = f"{name}_pic{i:04d}_{rand:0{digits}d}"

    img = Image.fromarray(generated_images[i]).resize((EXPORT_W, EXPORT_H), Image.LANCZOS)
    img.save(os.path.join(folder, f"{filename}.jpg"), quality=95)

    xc, yc, bw, bh = generated_labels_bbox[i]
    with open(os.path.join(folder, f"{filename}.txt"), "w") as f:
        f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")



print(f"\nTrain: {split_idx} Bilder → {train_folder}")
print(f"Test:  {len(generated_images) - split_idx} Bilder → {test_folder}")

print(f"\nImages: {generated_images.shape}")
print(f"BBox:   {generated_labels_bbox.shape}")
print(f"Class:  {generated_labels_class.shape}")

print(f"\n{len(generated_images)} synthetische Bilder generiert")
print(f"{len(bg_canvases)} Canvases erstellt | Originale unverändert: {len(backgrounds)}")
print(f"{len(backgrounds)} Hintergründe auf {BG_CANVAS_W}x{BG_CANVAS_H} zugeschnitten")

print(f"\n{len(backgrounds)} Hintergründe geladen")
print(f"\n{len(cards)} Karten geladen")

print(f"\nExport: {EXPORT_W}x{EXPORT_H} | Ratio: {OUTPUT_RATIO[0]}:{OUTPUT_RATIO[1]}")


plt.figure(figsize=(12, 16))
for i in range(min(12, len(generated_images))):
    ax = plt.subplot(4, 3, i + 1)
    ax.imshow(generated_images[i])

    # BBox zeichnen
    xc, yc, bw, bh = generated_labels_bbox[i]
    x = (xc - bw / 2) * BG_CANVAS_W
    y = (yc - bh / 2) * BG_CANVAS_H
    ax.add_patch(patches.Rectangle(
        (x, y), bw * BG_CANVAS_W, bh * BG_CANVAS_H,
        linewidth=2, edgecolor='lime', facecolor='none'
    ))

    label = label_names[generated_labels_class[i]]
    ax.set_title(f"Klasse {label}", fontsize=8)
    ax.axis('off')

plt.suptitle(f"Synthetische Daten ({EXPORT_W}x{EXPORT_H} | {OUTPUT_RATIO[0]}:{OUTPUT_RATIO[1]})", fontsize=14)
plt.tight_layout()
plt.show()