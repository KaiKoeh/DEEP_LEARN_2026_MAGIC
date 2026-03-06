import requests
import os
import time
import unicodedata
from helper_classes.config_loader import ConfigLoader

### CONFIG-LOADER
main_folder = os.path.dirname(os.path.abspath(__file__)) + "/"
config = ConfigLoader(main_folder + "config_file.txt")

set_code = "ltr"  # ice - Ice Age

# Scryfall Download Ordner
output_folder = config.scryfall_cards_path

# Foto-Verzeichnisse für echte Fotos
photo_base = config.photos_sorted_path

# Label Path
label_path = config.label_file_path

## Add Directory
os.makedirs(output_folder, exist_ok=True)

# Alle Karten des Sets laden
url = f"https://api.scryfall.com/cards/search?q=set:{set_code}&unique=cards"
all_cards = []
labels = {}

while url:
    response = requests.get(url).json()
    all_cards.extend(response["data"])
    url = response.get("next_page")
    time.sleep(0.1)

print(f"{len(all_cards)} Karten gefunden")

# Bestehende Labels laden
if os.path.exists(label_path):
    with open(label_path) as f:
        for i, line in enumerate(f.readlines()):
            labels[line.strip()] = i

def clean_name(name):
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = name.replace(" ", "-").replace("/", "-").replace(",", "").replace("'", "").lower()
    return name

for i, card in enumerate(all_cards):
    if "image_uris" not in card:
        continue

    img_url = card["image_uris"]["png"]
    name = f"{set_code}-{card['collector_number']}-{card['name']}"
    name = clean_name(name)

    file_path = os.path.join(output_folder, f"{name}.png")
    if os.path.exists(file_path):
        print(f"  [{i+1}/{len(all_cards)}] {name} → DATEI EXISTIERT")
    else:
        img_data = requests.get(img_url).content
        with open(file_path, "wb") as f:
            f.write(img_data)
        print(f"  [{i+1}/{len(all_cards)}] {name} → DOWNLOAD")
        time.sleep(0.1)

    if labels.get(name) is None:
        labels[name] = len(labels)
        print(f"    → Label hinzugefügt: {name} (ID {labels[name]})")

    photo_folder = os.path.join(photo_base, name)
    if not os.path.exists(photo_folder):
        os.makedirs(photo_folder, exist_ok=True)
        print(f"    → Foto-Verzeichnis angelegt: {name}")

    time.sleep(0.001)

with open(label_path, "w") as f:
    for label in labels:
        f.write(f"{label}\n")

print(f"Fertig! {len(labels)} Karten in {label_path}")
