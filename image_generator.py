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
import multiprocessing
from multiprocessing import Pool, cpu_count, Value
from helper_classes.config_loader import ConfigLoader

### CONFIG-LOADER
main_folder = os.path.dirname(os.path.abspath(__file__)) + "/"
config_loader = ConfigLoader(main_folder + "config_file.txt")

####### FILE-LOAD ########
bg_folder = main_folder + "image_generator_backgrounds"
card_folder = config_loader.scryfall_cards_path
label_file = config_loader.label_file_path

####### EXPORT ########
train_folder = config_loader.train_data_synthetic_path
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
BACKGROUND_VARIATIONS = 3 ## 2
CARDS_PER_CANVAS = 1 ## 2

### EXPORT DATA
DELETE_OLD_EXPORT = True


######## OUTPUT RATIO & SIZE ########
EXPORT_W = config_loader.width
EXPORT_H = config_loader.height

######## BACKGROUND SETUP
BG_CANVAS_LONG = 1024               # Längste Seite des Canvas
BG_EDGE_MARGIN = 0.20                # 20% Rand zur Kante beibehalten, Shift darf 80% nutzen
BG_ROTATE_RANGE = 90                # max Grad Rotation in beide Richtungen
BG_ZOOM_RANGE = (1.2, 1.8)


######## KARTE HINZUFÜGEN ----
CARD_SCALE_RANGE = (0.6, 0.9)       # Karte nimmt xx-xx% des Canvas ein (bezogen auf Höhe)
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

### BERECHNUNG RATIO SIZE & CANVAS
BG_CANVAS_W = int(BG_CANVAS_LONG * EXPORT_W / EXPORT_H)
BG_CANVAS_H = BG_CANVAS_LONG

### MULTIPROCESSING
NUM_WORKERS = max(1, cpu_count() - 2)

### Shared Counter für Progress-Logging
import threading
counter = None
total_tasks = 0
thread_counter = 0
thread_lock = threading.Lock()

def init_worker(shared_counter, shared_total):
    global counter, total_tasks
    counter = shared_counter
    total_tasks = shared_total


def add_camera_noise(image, intensity=0.10):
    noise = np.random.normal(0, intensity * 255, image.shape)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def add_color_tint(image):
    tint_strength = random.uniform(*ENTITY_COLOR_TINT)
    tint_color = random.choice([
            (1.0, 0.3, 0.3), # Rot
            (0.3, 0.3, 1.0), # Blau
            (1.0, 0.8, 0.3), # Warmweiß/Gelb
            (0.5, 0.8, 1.0), # Kaltweiß/Cyan
    ])
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
    kernel = np.zeros((size, size))
    kernel[size // 2, :] = 1.0 / size
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
        contrast = random.uniform(*ENTITY_CONTRAST)
        saturation = random.uniform(*ENTITY_SATURATION)
    else:
        brightness = random.uniform(*OVERALL_BRIGHTNESS)
        contrast = random.uniform(*OVERALL_CONTRAST)
        saturation = random.uniform(*OVERALL_SATURATION)

    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Color(img).enhance(saturation)
    return np.array(img)


##### WORKER: 1 Bild generieren + speichern
def generate_single_image(task):
    try:
        task_idx, ci, v, card_idx, folder = task

        canvas = bg_canvases[ci]
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
            return False  # Übersprungen

        pos_x = random.randint(pad_x, max_x)
        pos_y = random.randint(pad_y, max_y)

        result_np = augment_color(canvas.copy(), entity=True)
        result = Image.fromarray(result_np)
        result.paste(card_rotated, (pos_x, pos_y), card_rotated.convert("RGBA").split()[3])

        # BBox
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

        # 3) Reinzoomen + Center Crop
        zoomed_w = int(BG_CANVAS_W * PERSPECTIVE_ZOOM)
        zoomed_h = int(BG_CANVAS_H * PERSPECTIVE_ZOOM)
        zoomed = np.array(Image.fromarray(warped).resize((zoomed_w, zoomed_h), Image.LANCZOS))

        left = (zoomed_w - BG_CANVAS_W) // 2
        top = (zoomed_h - BG_CANVAS_H) // 2
        cropped = zoomed[top:top + BG_CANVAS_H, left:left + BG_CANVAS_W]

        # 4) BBox transformieren
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

        # 5) Gesamteffekte
        cropped = add_camera_noise(cropped, random.uniform(*NOISE_RANDOM))
        cropped = add_motion_blur(cropped, int(random.uniform(*MOTION_BLUR)))
        cropped = augment_color(cropped, False)

        # 6) Speichern
        name = label_names[cls]
        rand = random.randint(1, RANDOM_MAX)
        digits = len(str(RANDOM_MAX))
        filename = f"{name}_pic{task_idx:06d}_{rand:0{digits}d}"

        img = Image.fromarray(cropped).resize((EXPORT_W, EXPORT_H), Image.LANCZOS)
        img.save(os.path.join(folder, f"{filename}.jpg"), quality=95)

        bx, by, bbw, bbh = bbox
        with open(os.path.join(folder, f"{filename}.txt"), "w") as f:
            f.write(f"{cls} {bx:.6f} {by:.6f} {bbw:.6f} {bbh:.6f}")

        # Progress-Logging
        if counter is not None:
            # Multiprocessing (Mac/Linux)
            with counter.get_lock():
                counter.value += 1
                if counter.value % 100 == 0 or counter.value == total_tasks:
                    print(f"  Progress: {counter.value}/{total_tasks} ({counter.value/total_tasks*100:.1f}%)", flush=True)
        else:
            # ThreadPool (Windows)
            global thread_counter
            with thread_lock:
                thread_counter += 1
                if thread_counter % 100 == 0:
                    print(f"  Progress: {thread_counter}/{total_tasks} ({thread_counter/total_tasks*100:.1f}%)", flush=True)

        return True

    except Exception as e:
        print(f"  ERROR bei Task {task[0]}: {e}", flush=True)
        return False


############ MAIN
if __name__ == "__main__":
    import platform

    if platform.system() != 'Windows':
        multiprocessing.set_start_method('fork')

    print(f"Canvas-Größe: {BG_CANVAS_W}x{BG_CANVAS_H}")
    print(f"Export-Größe: {EXPORT_W}x{EXPORT_H}")
    print(f"Workers: {NUM_WORKERS}")

    #### Label_Names aus der Datei laden
    with open(label_file) as f:
        for i, name in enumerate(f.readlines()):
            label_names[i] = name.strip()

    for filename in sorted(os.listdir(card_folder)):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            card_name = filename.split(".")[0]
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


    ############ VORBEREITUNG: Ordner löschen/erstellen
    if DELETE_OLD_EXPORT:
        if os.path.exists(train_folder):
            shutil.rmtree(train_folder)
        if os.path.exists(test_folder):
            shutil.rmtree(test_folder)

    #### Background Prepair to CANVAS
    random.shuffle(backgrounds)
    for bg in backgrounds:
        h, w = bg.shape[:2]
        if w < BG_CANVAS_W or h < BG_CANVAS_H:
            scale = max(BG_CANVAS_W / w, BG_CANVAS_H / h)
            img = Image.fromarray(bg).resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            bg = np.array(img)
            h, w = bg.shape[:2]

        cx = (w - BG_CANVAS_W) // 2
        cy = (h - BG_CANVAS_H) // 2

        # Shift: 80% des verfügbaren Platzes nutzen, 20% Rand bleibt
        shift_range_x = max(1, int(cx * (1 - BG_EDGE_MARGIN)))
        shift_range_y = max(1, int(cy * (1 - BG_EDGE_MARGIN)))

        for v in range(BACKGROUND_VARIATIONS):
            shift_x = random.randint(-shift_range_x, shift_range_x)
            shift_y = random.randint(-shift_range_y, shift_range_y)
            left = max(0, min(cx + shift_x, w - BG_CANVAS_W))
            top = max(0, min(cy + shift_y, h - BG_CANVAS_H))
            crop = bg[top:top + BG_CANVAS_H, left:left + BG_CANVAS_W].copy()

            angle = random.uniform(-BG_ROTATE_RANGE, BG_ROTATE_RANGE)
            crop_img = Image.fromarray(crop).rotate(angle, resample=Image.BICUBIC, fillcolor=(0, 0, 0))

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


    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    ############ TASK-LISTE ERSTELLEN
    tasks = []
    task_idx = 0
    total_expected = len(bg_canvases) * CARDS_PER_CANVAS * len(cards)
    split_idx = int(total_expected * train_split_value)

    for ci in range(len(bg_canvases)):
        for v in range(CARDS_PER_CANVAS):
            for card_idx in range(len(cards)):
                folder = train_folder if task_idx < split_idx else test_folder
                tasks.append((task_idx, ci, v, card_idx, folder))
                task_idx += 1

    print(f"\nErwartete Bilder: {len(tasks)} (Train: {split_idx}, Test: {len(tasks) - split_idx})")
    print(f"Starte {NUM_WORKERS} Worker...")

    ############ GENERIERUNG (ThreadPool auf Windows, Single-Thread auf Mac)
    total_tasks = len(tasks)

    if platform.system() == 'Windows':
        from concurrent.futures import ThreadPoolExecutor
        print(f"Windows: ThreadPool mit {NUM_WORKERS} Threads")
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            results = list(executor.map(generate_single_image, tasks))
    else:
        print("Mac/Linux: Single-Thread Modus")
        results = []
        for i, task in enumerate(tasks):
            if i % 100 == 0:
                print(f"  Progress: {i}/{total_tasks} ({i/total_tasks*100:.1f}%)", flush=True)
            results.append(generate_single_image(task))
        print(f"  Progress: {total_tasks}/{total_tasks} (100.0%)")

    generated = sum(1 for r in results if r)
    skipped = sum(1 for r in results if not r)

    ############ ERGEBNIS
    print(f"\nGeneriert: {generated} | Übersprungen: {skipped}")

    train_count = len([f for f in os.listdir(train_folder) if f.endswith(".jpg")])
    test_count = len([f for f in os.listdir(test_folder) if f.endswith(".jpg")])
    print(f"Train: {train_count} Bilder → {train_folder}")
    print(f"Test:  {test_count} Bilder → {test_folder}")

    print(f"Export Size: {EXPORT_W}x{EXPORT_H}")

    ############ PLOT: 5x5 Random Bilder aus Train-Ordner laden
    PLOT_COLS = 5
    PLOT_ROWS = 5
    PLOT_COUNT = PLOT_COLS * PLOT_ROWS

    txt_files = [f for f in os.listdir(train_folder) if f.endswith(".txt")]
    random.shuffle(txt_files)
    plot_files = txt_files[:PLOT_COUNT]

    fig, axes = plt.subplots(PLOT_ROWS, PLOT_COLS, figsize=(15, 20))

    for i, txt_file in enumerate(plot_files):
        row = i // PLOT_COLS
        col = i % PLOT_COLS
        ax = axes[row][col]

        jpg_file = txt_file.replace(".txt", ".jpg")
        img = Image.open(os.path.join(train_folder, jpg_file))
        ax.imshow(img)

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

    for i in range(len(plot_files), PLOT_ROWS * PLOT_COLS):
        axes[i // PLOT_COLS][i % PLOT_COLS].axis('off')

    plt.suptitle(f"Synthetische Daten ({EXPORT_W}x{EXPORT_H}) — {generated} Bilder generiert", fontsize=14)
    plt.tight_layout()
    plt.show()
