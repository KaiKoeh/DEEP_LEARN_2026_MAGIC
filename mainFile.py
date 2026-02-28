import pandas as pd
import numpy as np
from PIL import Image


pic_main_resolution=512
pic_target_resolution=256 ### CAN BE CHANGED!!!!
pic_scale_factor = pic_main_resolution/pic_target_resolution


# === CSV laden ===
df = pd.read_csv("label_data/labels_my-project-name_2026-02-27-07-20-41.csv")




# === Bildpfade bauen ===
##img_base = "pfad/zu/photos_512"
##df["folder"] = df["image_name"].str.rsplit("_", n=1).str[0]
##df["filepath"] = img_base + "/" + df["folder"] + "/" + df["image_name"]

# === Bilder laden ===
##images = []
##for path in df["filepath"]:
  ##  img = Image.open(path)
 ##   images.append(np.array(img))

##X = np.array(images)  # Shape: (367, 512, 512, 3)