#!/usr/bin/env python3
import os
from PIL import Image

source_dir = "/Users/kaikohrsen/Documents/schulung/PythonWeekly/deep_learn_project/img_source/photos_finished"

jpg_files = sorted([f for f in os.listdir(source_dir) if f.lower().endswith(".jpg")])
print(f"{len(jpg_files)} Bilder gefunden")

for filename in jpg_files:
    filepath = os.path.join(source_dir, filename)
    img = Image.open(filepath)
    img = img.rotate(270, expand=True)
    img.save(filepath, quality=95)
    print(f"  {filename} rotiert")

print("\nFertig!")