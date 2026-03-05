import requests
import os
import time
import unicodedata

set_code = "ltr"  # ice - Ice Age

main_folder = os.path.dirname(os.path.abspath(__file__)) + "/"

# Images donload ordner für echte Fotos
output_folder = main_folder + "image_generator/cards"

# Foto-Verzeichnis für echte Fotos zum Kopieren
photo_base = main_folder + "img_source/photos"

# Label Path
label_path = main_folder + "label_file.txt"


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
    time.sleep(0.1)  # Scryfall Rate Limit: 10 req/sec

print(f"{len(all_cards)} Karten gefunden")

# Bestehende Labels laden
if os.path.exists(label_path):
    with open(label_path) as f:
        for i, line in enumerate(f.readlines()):
            labels[line.strip()] = i

def clean_name(name):
    # Sonderzeichen normalisieren (ó → o, é → e, etc.)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = name.replace(" ", "-").replace("/", "-").replace(",", "").replace("'", "").lower()
    return name

for i, card in enumerate(all_cards):

    if "image_uris" not in card:
        continue

    img_url = card["image_uris"]["png"]
    name = f"{set_code}-{card['collector_number']}-{card['name']}"
    name = clean_name(name)

    # Bild nur downloaden wenn Datei fehlt
    file_path = os.path.join(output_folder, f"{name}.png")
    if os.path.exists(file_path):
        print(f"  [{i+1}/{len(all_cards)}] {name} → DATEI EXISTIERT")
    else:
        img_data = requests.get(img_url).content
        with open(file_path, "wb") as f:
            f.write(img_data)
        print(f"  [{i+1}/{len(all_cards)}] {name} → DOWNLOAD")
        time.sleep(0.1)

    # Label IMMER prüfen und hinzufügen
    if labels.get(name) is None:
        labels[name] = len(labels)
        print(f"    → Label hinzugefügt: {name} (ID {labels[name]})")

    # Foto-Ordner erstellen falls nicht vorhanden
    photo_folder = os.path.join(photo_base, name)
    if not os.path.exists(photo_folder):
        os.makedirs(photo_folder, exist_ok=True)
        print(f"    → Foto-Verzeichnis angelegt: {name} ")

    time.sleep(0.001)

# Am Ende Labels speichern
with open(label_path, "w") as f:
    for label in labels:
        f.write(f"{label}\n")

print(f"Fertig! {len(labels)} Karten in label_file.txt")