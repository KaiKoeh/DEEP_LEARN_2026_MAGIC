
import os
import random
from PIL import Image, ImageOps
from config_loader import ConfigLoader

main_folder = os.path.dirname(os.path.abspath(__file__)) + "/"
source_dir = main_folder + "img_source/1_photos_sorted"
target_dir = main_folder + "img_source/2_photos_finished"

######## OUTPUT SIZE AUS CONFIG ########
config = ConfigLoader(main_folder + "config_file.txt")
EXPORT_W = config.width
EXPORT_H = config.height

### FILE RANDOMIZER
RANDOM_MAX = 9999

print(f"Zielgröße: {EXPORT_W}x{EXPORT_H}")

os.makedirs(target_dir, exist_ok=True)

for folder in sorted(os.listdir(source_dir)):
    source_folder = os.path.join(source_dir, folder)
    if not os.path.isdir(source_folder):
        continue

    jpg_files = sorted([f for f in os.listdir(source_folder) if f.lower().endswith((".jpg", ".jpeg"))])
    if not jpg_files:
        continue

    print(f"\n{folder}/ ({len(jpg_files)} Bilder)")

    for i, filename in enumerate(jpg_files, start=1):
        img = Image.open(os.path.join(source_folder, filename))
        img = ImageOps.exif_transpose(img)  # EXIF-Rotation anwenden
        w, h = img.size

        # Längste Seite auf passende Größe skalieren, dann Center-Crop
        scale = max(EXPORT_W / w, EXPORT_H / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        left = (new_w - EXPORT_W) // 2
        top = (new_h - EXPORT_H) // 2
        img = img.crop((left, top, left + EXPORT_W, top + EXPORT_H))

        # Name: foldername_pic01587.jpg (01 + random 1-999)

        rand = random.randint(1, RANDOM_MAX)
        digits = len(str(RANDOM_MAX))
        new_name = f"{folder}_pic{i:04d}_{rand:0{digits}d}.jpg"
        out_path = os.path.join(target_dir, new_name)

        img.save(out_path, quality=95)
        os.remove(os.path.join(source_folder, filename))  # Original löschen

        print(f"  {filename} ({w}x{h}) -> {new_name} ({EXPORT_W}x{EXPORT_H})")

print("\nFertig!")