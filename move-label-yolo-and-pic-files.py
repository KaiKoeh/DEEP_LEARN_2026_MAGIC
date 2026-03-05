import os

main_folder = "/Users/kaikohrsen/Documents/schulung/PythonWeekly/deep_learn_project/"
label_file = main_folder + "label_file.txt"
photos_finished = main_folder + "img_source/photos_finished"

# 1) Label-Datei einlesen
label_names = {}
with open(label_file) as f:
    for i, line in enumerate(f.readlines()):
        label_names[line.strip()] = i

print(f"{len(label_names)} Labels geladen")

# 2) Alle YOLO txt-Dateien durchgehen
txt_files = sorted([f for f in os.listdir(photos_finished) if f.endswith(".txt")])

updated = 0
no_image = 0
not_found = 0

for txt_file in txt_files:
    txt_path = os.path.join(photos_finished, txt_file)

    # Prüfen ob das Bild dazu existiert
    jpg_file = txt_file.replace(".txt", ".jpg")
    jpg_path = os.path.join(photos_finished, jpg_file)
    if not os.path.isfile(jpg_path):
        print(f"  ✗ Kein Bild gefunden: {jpg_file}")
        no_image += 1
        continue

    # YOLO-Datei lesen
    with open(txt_path) as f:
        content = f.read().strip()

    # Kartenname aus Dateiname extrahieren
    name = txt_file.rsplit("_pic", 1)[0]

    # Label-ID nachschlagen
    class_id = label_names.get(name)

    if class_id is None:
        print(f"  ✗ {name} nicht in label_file.txt gefunden!")
        not_found += 1
        continue

    # Erste Zahl (alte ID) durch korrekte ID ersetzen, BBox beibehalten
    parts = content.split(" ", 1)
    bbox = parts[1] if len(parts) > 1 else ""

    with open(txt_path, "w") as f:
        if bbox:
            f.write(f"{class_id} {bbox}")
        else:
            f.write(f"{class_id}")

    updated += 1
    print(f"  ✓ {txt_file} → ID {class_id} ({name})")

print(f"\nFertig! {updated} aktualisiert, {no_image} ohne Bild, {not_found} nicht in Labels")
