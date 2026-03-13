import os
import shutil
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from helper_classes.config_loader import ConfigLoader

main_folder = os.path.dirname(os.path.abspath(__file__)) + "/"
config = ConfigLoader(main_folder + "config_file.txt")

photo_raw = config.photo_raw_path
photo_skip = config.photo_raw_skip_path
photo_sorted = config.photos_sorted_path
label_file = config.label_file_path

# Label laden und fehlende Ordner erstellen
with open(label_file) as f:
    for line in f.readlines():
        name = line.strip()
        folder_path = os.path.join(photo_sorted, name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"  Ordner erstellt: {name}")

# Verfügbare Karten-Ordner laden
available_cards = sorted([d for d in os.listdir(photo_sorted)
                          if os.path.isdir(os.path.join(photo_sorted, d)) and not d.startswith(".")])

# Set-Code abfragen
set_code = input("Set-Code eingeben (z.B. 'ltr', 'akh', 'kld'): ").strip().lower()

#   Nummer → Ordnername, gefiltert nach Set
number_lookup = {}
for card in available_cards:
    parts = card.split("-")
    if len(parts) >= 2 and parts[0] == set_code:
        number = parts[1]
        number_lookup[number] = card

print(f"{len(number_lookup)} Karten für Set '{set_code}' verfügbar")
print("=" * 60)

# Alle Fotos laden
photos = sorted([f for f in os.listdir(photo_raw)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))])

photoLen = len(photos)
print(f"{photoLen} Fotos zum Sortieren\n")

already_sorted = 0
newly_sorted = 0
skipped = 0

for i, photo in enumerate(photos):
    photo_path = os.path.join(photo_raw, photo)

    img = Image.open(photo_path)
    img = ImageOps.exif_transpose(img)
    plt.figure(figsize=(8, 10))
    plt.imshow(img)
    plt.title(f"[{i+1}/{photoLen}] {photo}", fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)

    print(f"[{i+1}/{photoLen}] {photo}")
    print("-" * 40)

    while True:
        user_input = input("Nummer eingeben (oder 'skip' / 'quit'): ").strip().lower()

        if user_input == "quit":
            print(f"Abgebrochen. {newly_sorted} Fotos neu sortiert, {skipped} übersprungen.")
            plt.close('all')
            exit()

        if user_input == "skip":
            os.makedirs(photo_skip, exist_ok=True)
            shutil.move(photo_path, os.path.join(photo_skip, photo))
            if os.path.isfile(photo_path):
                os.remove(photo_path)
            skipped += 1
            print(f"  → verschoben nach photo_raw_skip/")
            break

        card_id = number_lookup.get(user_input)

        if card_id:
            target_dir = os.path.join(photo_sorted, card_id)
            target_path = os.path.join(target_dir, photo)
            if os.path.exists(target_path):
                print(f"  → bereits vorhanden in {card_id}/")
                already_sorted += 1
            else:
                shutil.move(photo_path, target_path)
                newly_sorted += 1
                print(f"  ✓ → {card_id}/")
            if os.path.isfile(photo_path):
                os.remove(photo_path)
            break
        else:
            print(f"  ✗ Nummer: '{user_input}' nicht gefunden!!!")

    plt.close('all')

print(f"{'=' * 60}")
print(f"Fertig! {newly_sorted} neu sortiert, {already_sorted} bereits vorhanden, {skipped} übersprungen.")
