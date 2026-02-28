#!/usr/bin/env python3
"""
Benennt JPGs um, skaliert auf 512x512 (Center-Crop) und speichert
alles flach in einem Zielordner. Zufallszahl verhindert Duplikate.
"""

import os
import random
from PIL import Image

source_dir = "/Users/kaikohrsen/Documents/schulung/PythonWeekly/deep_learn_project/img_source/photos"
target_dir = "/Users/kaikohrsen/Documents/schulung/PythonWeekly/deep_learn_project/img_source/photos_finished"
TARGET_SIZE = 512

os.makedirs(target_dir, exist_ok=True)

for folder in sorted(os.listdir(source_dir)):
    source_folder = os.path.join(source_dir, folder)
    if not os.path.isdir(source_folder):
        continue

    jpg_files = sorted([f for f in os.listdir(source_folder) if f.lower().endswith(".jpg")])
    if not jpg_files:
        continue

    print(f"\n{folder}/ ({len(jpg_files)} Bilder)")

    for i, filename in enumerate(jpg_files, start=1):
        img = Image.open(os.path.join(source_folder, filename))
        w, h = img.size

        # Kürzere Seite auf 512 skalieren, dann Center-Crop
        scale = TARGET_SIZE / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        left = (new_w - TARGET_SIZE) // 2
        top = (new_h - TARGET_SIZE) // 2
        img = img.crop((left, top, left + TARGET_SIZE, top + TARGET_SIZE))

        # Name: foldername_pic01587.jpg (01 + random 1-999)
        rand = random.randint(1, 999)
        new_name = f"{folder}_pic{i:02d}{rand}.jpg"
        out_path = os.path.join(target_dir, new_name)

        img.save(out_path, quality=95)
        os.remove(os.path.join(source_folder, filename))  # Original löschen

        print(f"  {filename} ({w}x{h}) -> {new_name}")

print("\nFertig!")