
import os
from helper_classes.config_loader import ConfigLoader
import shutil
import random


TEST_DATA_PERCENTAGE_PER_CARD = 0.2

### CONFIG-LOADER
main_folder = os.path.dirname(os.path.abspath(__file__)) + "/"
config = ConfigLoader(main_folder + "config_file.txt")

### Folders
real_data_train_path = config.train_data_real_path
real_data_test_path = config.test_data_real_path

### Create Folders
os.makedirs(real_data_train_path, exist_ok=True)
os.makedirs(real_data_test_path, exist_ok=True)

for filename in os.listdir(real_data_test_path):
    src = os.path.join(real_data_test_path, filename)
    dst = os.path.join(real_data_train_path, filename)
    shutil.copy2(src, dst)


file_names = set()
for filename in os.listdir(real_data_train_path):
    name = os.path.splitext(filename)[0]  # "hallo.jpg" → "hallo"
    file_names.add(name)


file_names = list(file_names)

# Dateien nach Karten-ID (z.B. "neo-283", "ltr-236")
card_groups = {}
for name in file_names:
    parts = name.split("-")
    card_id = parts[0] + "-" + parts[1]

    if card_id not in card_groups:
        card_groups[card_id] = []
    card_groups[card_id].append(name)


# Pro Karte 20% für Test
test_data_names = []
for card_id, names in card_groups.items():
    random.shuffle(names)
    split_idx = max(1, int(len(names) * TEST_DATA_PERCENTAGE_PER_CARD))
    test_data_names.extend(names[:split_idx])

### Verschieben der Split-Data
moved = 0
for name in test_data_names:
    jpg_file = name + ".jpg"
    txt_file = name + ".txt"

    jpg_exists = os.path.exists(os.path.join(real_data_train_path, jpg_file))
    txt_exists = os.path.exists(os.path.join(real_data_train_path, txt_file))

    if jpg_exists and txt_exists:
        shutil.move(os.path.join(real_data_train_path, jpg_file), os.path.join(real_data_test_path, jpg_file))
        shutil.move(os.path.join(real_data_train_path, txt_file), os.path.join(real_data_test_path, txt_file))
        moved += 1
    else:
        print(f"  Fehler Datei nicht vorhanden von: {name} (jpg={jpg_exists}, txt={txt_exists})")

print(f"\nDatei Verschoben: {moved}")
