#!/usr/bin/env python3
"""
Benennt alle JPG-Dateien in Unterverzeichnissen um und verschiebt sie in ein Zielverzeichnis.

Nutzung:
  python3 rename_cards.py /pfad/source /pfad/target

Beispiel:
  python3 rename_cards.py ~/Desktop/karten ~/Desktop/karten_renamed

Ergebnis:
  source/akh-268-forest/IMG_2122.JPG -> target/akh-268-forest/akh-268-forest_pic01.jpg
  Originale werden nach dem Kopieren geloescht.
"""

import os
import sys
import shutil


source_dir = "/Users/kaikohrsen/Documents/schulung/PythonWeekly/deep_learn_project/img_source/photos"
target_dir = "/Users/kaikohrsen/Documents/schulung/PythonWeekly/deep_learn_project/img_data/photos"

if not os.path.isdir(source_dir):
    print(f"Fehler: Source-Verzeichnis '{source_dir}' existiert nicht.")
    sys.exit(1)

for folder in sorted(os.listdir(source_dir)):
    source_folder = os.path.join(source_dir, folder)

    if not os.path.isdir(source_folder):
        continue

    # Alle JPG-Dateien sammeln und sortieren
    jpg_files = sorted([f for f in os.listdir(source_folder) if f.upper().endswith(".JPG")])

    if not jpg_files:
        continue

    # Zielordner anlegen
    target_folder = os.path.join(target_dir, folder)
    os.makedirs(target_folder, exist_ok=True)

    print(f"\n{folder}/ ({len(jpg_files)} Bilder)")

    for i, filename in enumerate(jpg_files, start=1):
        old_path = os.path.join(source_folder, filename)
        new_name = f"{folder}_pic{i:02d}.jpg"
        new_path = os.path.join(target_folder, new_name)

        print(f"  {filename} -> {new_name}")
        shutil.copy2(old_path, new_path)
        os.remove(old_path)

print("\nFertig!")