#!/usr/bin/env python3
"""
Löscht alle JPG-Dateien in final_data, die kein zugehöriges TXT-Label haben.
"""

import os

data_dir = os.path.dirname(os.path.abspath(__file__))

jpg_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".jpg")]
deleted = 0

for jpg in sorted(jpg_files):
    txt = jpg.rsplit(".", 1)[0] + ".txt"
    if not os.path.exists(os.path.join(data_dir, txt)):
        os.remove(os.path.join(data_dir, jpg))
        print(f"Gelöscht: {jpg}")
        deleted += 1

print(f"\nFertig! {deleted} Bilder ohne Label gelöscht.")
