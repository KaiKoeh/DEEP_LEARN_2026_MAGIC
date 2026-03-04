import requests
import os
import time
import unicodedata

set_code = "ltr"  # ice - Ice Age

output_folder = "image_generator/cards"
os.makedirs(output_folder, exist_ok=True)

# Alle Karten des Sets laden
url = f"https://api.scryfall.com/cards/search?q=set:{set_code}&unique=cards"
all_cards = []

while url:
    response = requests.get(url).json()
    all_cards.extend(response["data"])
    url = response.get("next_page")
    time.sleep(0.1)  # Scryfall Rate Limit: 10 req/sec

print(f"{len(all_cards)} Karten gefunden")

# Bestehende Labels laden
labels = {}

if os.path.exists("label_file.txt"):
    with open("label_file.txt") as f:
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
    name =clean_name(name)

    if name in labels:
        print(f"  [{i+1}/{len(all_cards)}] {name} → ÜBERSPRUNGEN")
        continue

    img_data = requests.get(img_url).content
    with open(os.path.join(output_folder, f"{name}.png"), "wb") as f:
        f.write(img_data)

    # Prüfen ob Label schon exestiert und spreichern
    if labels.get(name) is None:
        labels[name] = len(labels)

    print(f"  [{i+1}/{len(all_cards)}] {name}")
    time.sleep(0.1)

# Am Ende Labels speichern
with open("label_file.txt", "w") as f:
    for label in labels:
        f.write(f"{label}\n")

print(f"Fertig! {len(labels)} Karten in label_file.txt")