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
##main_folder = "/Users/kaikohrsen/Documents/schulung/PythonWeekly/deep_learn_project/"
main_folder = r"C:\Users\MrKoiKoi\PycharmProjects\PythonProject\EndProjekt" + "\\"

####### FILE-LOAD ########
bg_folder = main_folder + "image_generator/backgrounds"
card_folder = main_folder + "image_generator/cards"
label_file = main_folder + "label_file.txt"


####### EXPORT ########
output_folder = main_folder + "final_data"
train_folder = output_folder + "/train_data_synthetic"
test_folder = output_folder + "/test_data_synthetic"

train_split_value = 0.8

label_names = {}
backgrounds = []
cards = []
card_classes = []

### Erzeugte Backgrounds
canvases = []

BACKGROUND_VARIATIONS = 9 ## 20
CARDS_PER_CANVAS = 30 ## 20


def add_camera_noise(image, intensity=0.10):
    """Künstliches Kamera-Rauschen (Gaussian Noise)"""
    noise = np.random.normal(0, intensity * 255, image.shape)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def add_motion_blur(image, max_kernel=7):
    """Simuliert Kamera-Verwackeln (Motion Blur)"""
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

def augment_color(image, soft=False):
    """Zufällige Farb-Augmentation: Brightness, Contrast, Saturation"""
    img = Image.fromarray(image)

    from PIL import ImageEnhance

    if soft:
        brightness = random.uniform(0.8, 1.2)
    else:
        brightness = random.uniform(0.6, 1.4)

    img = ImageEnhance.Brightness(img).enhance(brightness)

    if soft:
        contrast = random.uniform(0.6, 1.4)
    else:
        contrast = random.uniform(0.8, 1.2)

    img = ImageEnhance.Contrast(img).enhance(contrast)

    if soft:
        saturation = random.uniform(0.8, 1.2)
    else:
        saturation = random.uniform(0.5, 1.5)

    img = ImageEnhance.Color(img).enhance(saturation)

    return np.array(img)


#### Label_Names aus der Datei laden
with open(label_file) as f:
    for i, name in enumerate(f.readlines()):
        label_names[i] = name.strip()


for filename in sorted(os.listdir(card_folder)):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        img = Image.open(os.path.join(card_folder, filename))
        cards.append(np.array(img))

        card_name = filename.split(".")[0]

        ### Karten den Label-Ids zuweisen
        for class_id, label in label_names.items():
            if label == card_name:
                card_classes.append(class_id)
                print(f"  {filename} → Klasse {class_id} ({label})")
                break


for filename in sorted(os.listdir(bg_folder)):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        img = Image.open(os.path.join(bg_folder, filename))
        backgrounds.append(np.array(img))
        print(f"  {filename} → {img.size}")



######## BACKGROUNDS ----

CANVAS_SIZE = 1024 # GRÖßE DES BACKGROUNDS
SHIFT_RANGE = 150   # max Pixel Verschiebung
ROTATE_RANGE = 90   # max Grad Rotation
ZOOM_RANGE = (1.3, 3.0)

for bg in backgrounds:
    h, w = bg.shape[:2]

    if w < CANVAS_SIZE or h < CANVAS_SIZE:
        scale = CANVAS_SIZE / min(w, h)
        img = Image.fromarray(bg).resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        bg = np.array(img)
        h, w = bg.shape[:2]

    cx = (w - CANVAS_SIZE) // 2
    cy = (h - CANVAS_SIZE) // 2
    canvas = bg[cy:cy + CANVAS_SIZE, cx:cx + CANVAS_SIZE].copy()
    canvases.append(canvas)

    for v in range(BACKGROUND_VARIATIONS):
        shift_x = random.randint(-SHIFT_RANGE, SHIFT_RANGE)
        shift_y = random.randint(-SHIFT_RANGE, SHIFT_RANGE)

        left = max(0, min(cx + shift_x, w - CANVAS_SIZE))
        top = max(0, min(cy + shift_y, h - CANVAS_SIZE))


        crop = bg[top:top + CANVAS_SIZE, left:left + CANVAS_SIZE].copy()
        crop = augment_color(crop, soft=True)

        # Rotieren
        angle = random.uniform(-ROTATE_RANGE, ROTATE_RANGE)
        crop_img = Image.fromarray(crop).rotate(angle, resample=Image.BICUBIC, fillcolor=(0, 0, 0))

        # Random Zoom: hochskalieren, dann Mitte 1024x1024 ausschneiden
        zoom = random.uniform(*ZOOM_RANGE)
        zoomed_size = int(CANVAS_SIZE * zoom)
        crop_img = crop_img.resize((zoomed_size, zoomed_size), Image.LANCZOS)

        zl = (zoomed_size - CANVAS_SIZE) // 2 + random.randint(-30, 30)
        zt = (zoomed_size - CANVAS_SIZE) // 2 + random.randint(-30, 30)
        zl = max(0, min(zl, zoomed_size - CANVAS_SIZE))
        zt = max(0, min(zt, zoomed_size - CANVAS_SIZE))

        crop_img = crop_img.crop((zl, zt, zl + CANVAS_SIZE, zt + CANVAS_SIZE))
        canvases.append(np.array(crop_img))



###### KARTE HINZUFÜGEN ----
CARD_SCALE_RANGE = (0.35, 0.7)   # Karte nimmt xxx-xxxx% des Canvas ein
CARD_ROTATE_RANGE = 45          # PROZENT Rotation
CARD_PADDING = 0.15  # xx% Abstand vom Rand

generated_images = []
generated_labels_bbox = []   # (xc, yc, w, h) normalisiert
generated_labels_class = []  # class_id

for canvas in canvases:
    for _ in range(CARDS_PER_CANVAS):
        card_idx = random.randint(0, len(cards) - 1)
        card = Image.fromarray(augment_color(cards[card_idx], soft=True))

        scale = random.uniform(*CARD_SCALE_RANGE)
        card_h = int(CANVAS_SIZE * scale)
        card_w = int(card_h * (card.width / card.height))
        card_resized = card.resize((card_w, card_h), Image.LANCZOS)

        angle = random.uniform(-CARD_ROTATE_RANGE, CARD_ROTATE_RANGE)
        card_rotated = card_resized.rotate(angle, expand=True, resample=Image.BICUBIC)
        rot_w, rot_h = card_rotated.size

        pad = int(CANVAS_SIZE * CARD_PADDING)
        max_x = CANVAS_SIZE - rot_w - pad
        max_y = CANVAS_SIZE - rot_h - pad

        if max_x < pad or max_y < pad:
            continue

        pos_x = random.randint(pad, max_x)
        pos_y = random.randint(pad, max_y)
        result = Image.fromarray(canvas.copy())
        result.paste(card_rotated, (pos_x, pos_y), card_rotated.convert("RGBA").split()[3])

        xc = (pos_x + rot_w / 2) / CANVAS_SIZE
        yc = (pos_y + rot_h / 2) / CANVAS_SIZE
        bw = rot_w / CANVAS_SIZE
        bh = rot_h / CANVAS_SIZE

        generated_images.append(np.array(result))
        generated_labels_bbox.append([xc, yc, bw, bh])
        generated_labels_class.append(card_classes[card_idx])  # ← Klasse aus label_file.txt

# In NumPy Arrays konvertieren
generated_images = np.array(generated_images)
generated_labels_bbox = np.array(generated_labels_bbox, dtype=np.float32)
generated_labels_class = np.array(generated_labels_class, dtype=np.int32)



######### KIPPEN DER PERSPEKTIVE + CUT OUT
PERSPECTIVE_SHIFT = 0.15
PERSPECTIVE_PAD = 0.2  # extra Rand zum Wegschneiden

for i in range(len(generated_images)):
    img = generated_images[i]

    # 1) Bild vergrößern → extra Rand für Perspektive
    pad_px = int(CANVAS_SIZE * PERSPECTIVE_PAD)
    big_size = CANVAS_SIZE + 2 * pad_px
    big_img = np.array(
        Image.fromarray(img).resize((big_size, big_size), Image.LANCZOS)
    )

    # 2) Perspektive kippen
    shift = int(big_size * PERSPECTIVE_SHIFT)
    pts1 = np.float32([[0, 0], [big_size, 0], [0, big_size], [big_size, big_size]])
    pts2 = np.float32([
        [random.randint(0, shift), random.randint(0, shift)],
        [big_size - random.randint(0, shift), random.randint(0, shift)],
        [random.randint(0, shift), big_size - random.randint(0, shift)],
        [big_size - random.randint(0, shift), big_size - random.randint(0, shift)]
    ])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(big_img, matrix, (big_size, big_size))

    # 3) Mitte ausschneiden → schwarze hoffentlich Ränder weg
    generated_images[i] = warped[pad_px:pad_px + CANVAS_SIZE, pad_px:pad_px + CANVAS_SIZE]

    # 4) BBox transformieren
    xc, yc, bw, bh = generated_labels_bbox[i]

    # BBox-Koordinaten auf big_size skalieren
    x1 = (xc - bw / 2) * big_size
    y1 = (yc - bh / 2) * big_size
    x2 = (xc + bw / 2) * big_size
    y2 = (yc + bh / 2) * big_size

    corners = np.float32([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
    corners = corners.reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(corners, matrix)
    transformed = transformed.reshape(-1, 2)

    # Auf Crop-Bereich umrechnen (pad_px abziehen)
    new_x1 = max(0, transformed[:, 0].min() - pad_px)
    new_y1 = max(0, transformed[:, 1].min() - pad_px)
    new_x2 = min(CANVAS_SIZE, transformed[:, 0].max() - pad_px)
    new_y2 = min(CANVAS_SIZE, transformed[:, 1].max() - pad_px)

    generated_labels_bbox[i] = [
        (new_x1 + new_x2) / 2 / CANVAS_SIZE,
        (new_y1 + new_y2) / 2 / CANVAS_SIZE,
        (new_x2 - new_x1) / CANVAS_SIZE,
        (new_y2 - new_y1) / CANVAS_SIZE
    ]


NOISE_RANDOM = (0.02, 0.10)
MOTION_BLUR = (3, 15)

for i in range(len(generated_images)):
    noise_random  = random.uniform(*NOISE_RANDOM)
    generated_images[i] = add_camera_noise(generated_images[i], noise_random)
    blur_random = int(random.uniform(*MOTION_BLUR))
    generated_images[i] = add_motion_blur(generated_images[i], blur_random)
    generated_images[i] = augment_color(generated_images[i], True)


##### SHUFFLE

indices = np.arange(len(generated_images))
np.random.shuffle(indices)

generated_images = generated_images[indices]
generated_labels_bbox = generated_labels_bbox[indices]
generated_labels_class = generated_labels_class[indices]

##### EXPORT FILES
EXPORT_SIZE = 512


## Alte Daten löschen
if os.path.exists(train_folder):
    shutil.rmtree(train_folder)
if os.path.exists(test_folder):
    shutil.rmtree(test_folder)

os.makedirs(train_folder)
os.makedirs(test_folder)


split_idx = int(len(generated_images) * train_split_value)

RANDOM_MAX = 9999

rand = random.randint(1, 999)

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

    img = Image.fromarray(generated_images[i]).resize((EXPORT_SIZE, EXPORT_SIZE), Image.LANCZOS)
    img.save(os.path.join(folder, f"{filename}.jpg"), quality=95)

    xc, yc, bw, bh = generated_labels_bbox[i]
    with open(os.path.join(folder, f"{filename}.txt"), "w") as f:
        f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")



print(f"Train: {split_idx} Bilder → {train_folder}")
print(f"Test:  {len(generated_images) - split_idx} Bilder → {test_folder}")


print(f"Images: {generated_images.shape}")
print(f"BBox:   {generated_labels_bbox.shape}")
print(f"Class:  {generated_labels_class.shape}")

print(f"{len(generated_images)} synthetische Bilder generiert")

print(f"{len(canvases)} Canvases erstellt | Originale unverändert: {len(backgrounds)}")
print(f"{len(backgrounds)} Hintergründe auf {CANVAS_SIZE}x{CANVAS_SIZE} zugeschnitten")


print(f"\n{len(backgrounds)} Hintergründe geladen")
print(f"\n{len(cards)} Hintergründe geladen")


plt.figure(figsize=(18, 12))
for i in range(min(12, len(generated_images))):
    ax = plt.subplot(3, 4, i + 1)
    ax.imshow(generated_images[i])

    # BBox zeichnen
    xc, yc, bw, bh = generated_labels_bbox[i]
    x = (xc - bw / 2) * CANVAS_SIZE
    y = (yc - bh / 2) * CANVAS_SIZE
    ax.add_patch(patches.Rectangle(
        (x, y), bw * CANVAS_SIZE, bh * CANVAS_SIZE,
        linewidth=2, edgecolor='lime', facecolor='none'
    ))


    label = label_names[generated_labels_class[i]]
    ax.set_title(f"Klasse {label}", fontsize=8)
    ax.axis('off')

plt.suptitle("Synthetische Daten mit Auto-Labels", fontsize=14)
plt.tight_layout()
plt.show()