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
from helper_classes.config_loader import ConfigLoader

### CONFIG-LOADER
main_folder = os.path.dirname(os.path.abspath(__file__)) + "/"
config_loader = ConfigLoader(main_folder + "config_file.txt")

####### FILE-LOAD ########
bg_folder = main_folder + "image_generator_backgrounds"
card_folder = config_loader.scryfall_cards_path
label_file = config_loader.label_file_path

####### EXPORT ########
train_folder = config_loader.train_data_path
test_folder = config_loader.test_data_synthetic_path

###### TRAIN-DATA PER %
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
BACKGROUND_VARIATIONS = 2 ## 20
CARDS_PER_CANVAS = 4 ## 20

### EXPORT DATA
DELETE_OLD_EXPORT = True


######## OUTPUT RATIO & SIZE ########
EXPORT_W = config_loader.width
EXPORT_H = config_loader.height

######## BACKGROUND SETUP
BG_CANVAS_LONG = 1024               # Längste Seite des Canvas
BG_SHIFT_RANGE = 150                # max Pixel Verschiebung
BG_ROTATE_RANGE = 90                # max Grad Rotation in beide Richtungen
BG_ZOOM_RANGE = (1.2, 1.8)


######## KARTE HINZUFÜGEN ----
CARD_SCALE_RANGE = (0.6, 0.9)      # Karte nimmt xx-xx% des Canvas ein (bezogen auf Höhe)
CARD_ROTATE_RANGE = 35              # PROZENT Rotation
CARD_PADDING = 0.15                 # xx% Abstand vom Rand der Karten position, ACHTUNG KIPPEN UND PAD Schnitt testen!
PERSPECTIVE_SHIFT = 0.125            # MAX KIPP WINKEL DER PERSPEKTIVE

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
ENTITY_COLOR_TINT = (0.08, 0.3)    ### TINT COLOR > CARD

### INTERNAL CALCULATION
PERSPECTIVE_ZOOM = 1.7
CARD_SCALE_SHRINK = 0.5
RANDOM_MAX = 9999


def add_camera_noise(image, intensity=0.10):
    noise = np.random.normal(0, intensity * 255, image.shape)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def add_color_tint(image):
    tint_strength = random.uniform(*ENTITY_COLOR_TINT)
    tint_color = random.choice([
        (1.0, 0.3, 0.3),  # Rot
        (0.3, 0.3, 1.0),  # Blau
        (1.0, 0.8, 0.3),  # Warmweiß/Gelb
        (0.5, 0.8, 1.0),  # Kaltweiß/Cyan
    ])

    # Nur RGB-Kanäle tinten, Alpha beibehalten
    rgb = image[:, :, :3].astype(np.float32)
    tint = np.array(tint_color) * 255
    rgb = rgb * (1 - tint_strength) + tint * tint_strength

    result = image.copy()
    result[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)

    return result

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

BG_CANVAS_W = int(BG_CANVAS_LONG * EXPORT_W / EXPORT_H)
BG_CANVAS_H = BG_CANVAS_LONG

print(f"Canvas-Größe: {BG_CANVAS_W}x{BG_CANVAS_H}")
print(f"Export-Größe: {EXPORT_W}x{EXPORT_H}")


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
random.shuffle(backgrounds)
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

    for v in range(BACKGROUND_VARIATIONS):
        shift_x = random.randint(-BG_SHIFT_RANGE, BG_SHIFT_RANGE)
        shift_y = random.randint(-BG_SHIFT_RANGE, BG_SHIFT_RANGE)

        left = max(0, min(cx + shift_x, w - BG_CANVAS_W))
        top = max(0, min(cy + shift_y, h - BG_CANVAS_H))

        crop = bg[top:top + BG_CANVAS_H, left:left + BG_CANVAS_W].copy()

        # Rotieren
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


############ VORBEREITUNG: Ordner löschen/erstellen

if DELETE_OLD_EXPORT:
    if os.path.exists(train_folder):
        shutil.rmtree(train_folder)
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)


############ BERECHNUNG: Gesamtanzahl für Train/Test Split

total_expected = len(bg_canvases) * CARDS_PER_CANVAS * len(cards)
split_idx = int(total_expected * train_split_value)

print(f"\nErwartete Bilder: ~{total_expected} (Train: ~{split_idx}, Test: ~{total_expected - split_idx})")


############ GENERATE IMAGES + EFFEKTE + DIREKT SPEICHERN
print("GENERATE IMAGES")

canvas_amount = len(bg_canvases)
card_amount = len(cards)
skipped = 0
image_count = 0

total_images = canvas_amount * (card_amount * CARDS_PER_CANVAS)

for ci, canvas in enumerate(bg_canvases):
    for v in range(CARDS_PER_CANVAS):
        for card_idx in range(len(cards)):
            if image_count % 10 == 0:
                print(f"Progress: {image_count} / {total_images}")

            cardData = cards[card_idx].copy()

            if random.random() < 0.25:
                cardData = add_color_tint(cardData)

            card = Image.fromarray(cardData)

            # 1) Karte kleiner platzieren (SHRINK)
            scale = random.uniform(*CARD_SCALE_RANGE) * CARD_SCALE_SHRINK
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
                skipped += 1
                continue

            pos_x = random.randint(pad_x, max_x)
            pos_y = random.randint(pad_y, max_y)

            result_np = augment_color(canvas.copy(), entity=True)
            result = Image.fromarray(result_np)
            result.paste(card_rotated, (pos_x, pos_y), card_rotated.convert("RGBA").split()[3])

            # BBox vor Perspektive (normalisiert)
            xc = (pos_x + rot_w / 2) / BG_CANVAS_W
            yc = (pos_y + rot_h / 2) / BG_CANVAS_H
            bw = rot_w / BG_CANVAS_W
            bh = rot_h / BG_CANVAS_H

            # 2) Perspektive kippen
            result_np = np.array(result)
            shift_w = int(BG_CANVAS_W * PERSPECTIVE_SHIFT)
            shift_h = int(BG_CANVAS_H * PERSPECTIVE_SHIFT)

            pts1 = np.float32([[0, 0], [BG_CANVAS_W, 0],
                               [0, BG_CANVAS_H], [BG_CANVAS_W, BG_CANVAS_H]])
            pts2 = np.float32([
                [random.randint(0, shift_w), random.randint(0, shift_h)],
                [BG_CANVAS_W - random.randint(0, shift_w), random.randint(0, shift_h)],
                [random.randint(0, shift_w), BG_CANVAS_H - random.randint(0, shift_h)],
                [BG_CANVAS_W - random.randint(0, shift_w), BG_CANVAS_H - random.randint(0, shift_h)]
            ])

            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            warped = cv2.warpPerspective(result_np, matrix, (BG_CANVAS_W, BG_CANVAS_H))

            # 3) Reinzoomen + Center Crop → schwarze Ränder weg
            zoomed_w = int(BG_CANVAS_W * PERSPECTIVE_ZOOM)
            zoomed_h = int(BG_CANVAS_H * PERSPECTIVE_ZOOM)
            zoomed = np.array(Image.fromarray(warped).resize((zoomed_w, zoomed_h), Image.LANCZOS))

            left = (zoomed_w - BG_CANVAS_W) // 2
            top = (zoomed_h - BG_CANVAS_H) // 2
            cropped = zoomed[top:top + BG_CANVAS_H, left:left + BG_CANVAS_W]

            # 4) BBox durch Perspektive + Zoom transformieren
            x1 = (xc - bw / 2) * BG_CANVAS_W
            y1 = (yc - bh / 2) * BG_CANVAS_H
            x2 = (xc + bw / 2) * BG_CANVAS_W
            y2 = (yc + bh / 2) * BG_CANVAS_H

            corners = np.float32([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
            corners = corners.reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(corners, matrix)
            transformed = transformed.reshape(-1, 2)

            transformed[:, 0] = transformed[:, 0] * PERSPECTIVE_ZOOM - left
            transformed[:, 1] = transformed[:, 1] * PERSPECTIVE_ZOOM - top

            new_x1 = max(0, transformed[:, 0].min())
            new_y1 = max(0, transformed[:, 1].min())
            new_x2 = min(BG_CANVAS_W, transformed[:, 0].max())
            new_y2 = min(BG_CANVAS_H, transformed[:, 1].max())

            bbox = [
                (new_x1 + new_x2) / 2 / BG_CANVAS_W,
                (new_y1 + new_y2) / 2 / BG_CANVAS_H,
                (new_x2 - new_x1) / BG_CANVAS_W,
                (new_y2 - new_y1) / BG_CANVAS_H
            ]
            cls = card_classes[card_idx]

            # 5) Gesamteffekte: Noise, Blur, Farbe
            noise_random = random.uniform(*NOISE_RANDOM)
            cropped = add_camera_noise(cropped, noise_random)
            blur_random = int(random.uniform(*MOTION_BLUR))
            cropped = add_motion_blur(cropped, blur_random)
            cropped = augment_color(cropped, False)

            # 6) Train/Test Split + Speichern
            if image_count < split_idx:
                folder = train_folder
            else:
                folder = test_folder

            name = label_names[cls]
            rand = random.randint(1, RANDOM_MAX)
            digits = len(str(RANDOM_MAX))
            filename = f"{name}_pic{image_count:04d}_{rand:0{digits}d}"

            img = Image.fromarray(cropped).resize((EXPORT_W, EXPORT_H), Image.LANCZOS)
            img.save(os.path.join(folder, f"{filename}.jpg"), quality=95)

            bx, by, bbw, bbh = bbox
            with open(os.path.join(folder, f"{filename}.txt"), "w") as f:
                f.write(f"{cls} {bx:.6f} {by:.6f} {bbw:.6f} {bbh:.6f}")

            image_count += 1


############ ERGEBNIS
print(f"\nGeneriert: {image_count} | Übersprungen: {skipped}")

train_count = min(image_count, split_idx)
test_count = image_count - train_count
print(f"Train: {train_count} Bilder → {train_folder}")
print(f"Test:  {test_count} Bilder → {test_folder}")

print(f"\n{len(bg_canvases)} Canvases erstellt | {len(backgrounds)} Hintergründe")
print(f"{len(cards)} Karten geladen")
print(f"Export: {EXPORT_W}x{EXPORT_H}")


############ PLOT: 5x5 Random Bilder aus Train-Ordner laden
PLOT_COLS = 5
PLOT_ROWS = 5
PLOT_COUNT = PLOT_COLS * PLOT_ROWS

txt_files = [f for f in os.listdir(train_folder) if f.endswith(".txt")]
random.shuffle(txt_files)

plot_files = random.sample(txt_files, min(PLOT_COUNT, len(txt_files)))

fig, axes = plt.subplots(PLOT_ROWS, PLOT_COLS, figsize=(15, 20))

for i, txt_file in enumerate(plot_files):
    row = i // PLOT_COLS
    col = i % PLOT_COLS
    ax = axes[row][col]

    # Bild laden
    jpg_file = txt_file.replace(".txt", ".jpg")
    img = Image.open(os.path.join(train_folder, jpg_file))
    ax.imshow(img)

    # BBox laden
    with open(os.path.join(train_folder, txt_file)) as f:
        parts = f.read().strip().split()
    cls = int(parts[0])
    xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

    x = (xc - bw / 2) * EXPORT_W
    y = (yc - bh / 2) * EXPORT_H
    ax.add_patch(patches.Rectangle(
        (x, y), bw * EXPORT_W, bh * EXPORT_H,
        linewidth=2, edgecolor='lime', facecolor='none'
    ))

    ax.set_title(label_names.get(cls, f"ID {cls}"), fontsize=7)
    ax.axis('off')

# Leere Felder ausblenden
for i in range(len(plot_files), PLOT_ROWS * PLOT_COLS):
    axes[i // PLOT_COLS][i % PLOT_COLS].axis('off')

plt.suptitle(f"Synthetische Daten ({EXPORT_W}x{EXPORT_H}) — {image_count} Bilder generiert", fontsize=14)
plt.tight_layout()
plt.show()
