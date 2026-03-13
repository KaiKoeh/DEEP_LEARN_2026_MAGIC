import os
import random
from PIL import Image, ImageOps
from helper_classes.config_loader import ConfigLoader

main_folder = os.path.dirname(os.path.abspath(__file__)) + "/"
config = ConfigLoader(main_folder + "config_file.txt")

source_dir = config.photos_sorted_path
target_dir = config.photos_finished_path

EXPORT_W = config.width
EXPORT_H = config.height

RANDOM_MAX = 9999

print(f"Zielgröße: {EXPORT_W}x{EXPORT_H}")

os.makedirs(target_dir, exist_ok=True)
data_list = os.listdir(source_dir)

for folder in sorted(data_list):
    source_folder = os.path.join(source_dir, folder)
    if not os.path.isdir(source_folder):
        continue

    source_list = os.listdir(source_folder)
    jpg_files = []

    for f in sorted(source_list):
        if f.lower().endswith((".jpg", ".jpeg")):
            jpg_files.append(f)

    if not jpg_files:
        continue

    print(f"{folder}/ ({len(jpg_files)} Bilder)")

    for i, filename in enumerate(jpg_files, start=1):
        img = Image.open(os.path.join(source_folder, filename))
        img = ImageOps.exif_transpose(img)
        w, h = img.size

        scale = max(EXPORT_W / w, EXPORT_H / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        left = (new_w - EXPORT_W) // 2
        top = (new_h - EXPORT_H) // 2
        img = img.crop((left, top, left + EXPORT_W, top + EXPORT_H))

        rand = random.randint(1, RANDOM_MAX)
        digits = len(str(RANDOM_MAX))
        new_name = f"{folder}_pic{i:04d}_{rand:0{digits}d}.jpg"
        out_path = os.path.join(target_dir, new_name)

        img.save(out_path, quality=95)
        os.remove(os.path.join(source_folder, filename))

        print(f"  {filename} ({w}x{h}) -> {new_name} ({EXPORT_W}x{EXPORT_H})")

print("\nFertig!")
