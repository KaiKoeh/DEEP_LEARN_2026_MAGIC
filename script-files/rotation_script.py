#!/usr/bin/env python3
"""
Rotiert alle JPG-Bilder 90 Grad gegen den Uhrzeigersinn,
damit die Karten aufrecht stehen (Symbol unten).
"""

import os
from PIL import Image

source_dir = "/Users/kaikohrsen/Documents/schulung/PythonWeekly/deep_learn_project/img_data/photos_512"

for folder in sorted(os.listdir(source_dir)):
    folder_path = os.path.join(source_dir, folder)

    if not os.path.isdir(folder_path):
        continue

    jpg_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")])

    if not jpg_files:
        continue

    print(f"\n{folder}/ ({len(jpg_files)} Bilder)")

    for filename in jpg_files:
        filepath = os.path.join(folder_path, filename)
        img = Image.open(filepath)
        img = img.rotate(-90, expand=True)
        img.save(filepath, quality=95)
        print(f"  {filename} rotiert")

print("\nFertig!")