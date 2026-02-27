#!/usr/bin/env python3
"""
Skaliert alle JPG-Bilder auf 512x512.
Strategie: Kuerzere Seite auf 512 skalieren, dann mittig croppen.
Kein Verzerren, kein Padding.

Nutzung: Pfade unten anpassen und ausfuehren.
"""

import os
from PIL import Image

source_dir = "/Users/kaikohrsen/Documents/schulung/PythonWeekly/deep_learn_project/img_data/photos"
target_dir = "/Users/kaikohrsen/Documents/schulung/PythonWeekly/deep_learn_project/img_data/photos_512"

TARGET_SIZE = 512

for folder in sorted(os.listdir(source_dir)):
    source_folder = os.path.join(source_dir, folder)

    if not os.path.isdir(source_folder):
        continue

    target_folder = os.path.join(target_dir, folder)
    os.makedirs(target_folder, exist_ok=True)

    jpg_files = sorted([f for f in os.listdir(source_folder) if f.lower().endswith(".jpg")])

    if not jpg_files:
        continue

    print(f"\n{folder}/ ({len(jpg_files)} Bilder)")

    for filename in jpg_files:
        img = Image.open(os.path.join(source_folder, filename))
        w, h = img.size

        # Kuerzere Seite auf TARGET_SIZE skalieren (Seitenverhaeltnis beibehalten)
        scale = TARGET_SIZE / min(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # Mittig auf 512x512 croppen
        left = (new_w - TARGET_SIZE) // 2
        top = (new_h - TARGET_SIZE) // 2
        img = img.crop((left, top, left + TARGET_SIZE, top + TARGET_SIZE))

        out_path = os.path.join(target_folder, filename)
        img.save(out_path, quality=95)
        print(f"  {filename} ({w}x{h}) -> {TARGET_SIZE}x{TARGET_SIZE}")

print("\nFertig!")