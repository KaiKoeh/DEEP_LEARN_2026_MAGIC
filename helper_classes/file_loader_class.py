import os
import numpy as np
from PIL import Image

class FileLoader:

    def __init__(self, folder, label_folder, target_width_resolution, target_height_resolution):

        #### params
        self.folder = folder
        self.target_width_resolution = target_width_resolution
        self.target_height_resolution = target_height_resolution

        #### Empty Vars
        self.images = None
        self.y_bbox = None
        self.y_class = None
        self.label_names = {}

        #### Load Label-Data
        self.load_label_data(label_folder)


    def count_files(self, file_ending):
        amount = 0
        for f in os.listdir(self.folder):
            if f.endswith(file_ending):
                amount += 1
        return amount

    def load_label_data(self, label_path):
        print("Loading label file: " + label_path)
        self.label_names = {}
        with open(label_path) as f:
            for i, line in enumerate(f.readlines()):
                self.label_names[i] = line.strip()
        return self

    def load(self):

        ### LADEN DER DATEI COUNT
        txt_count = self.count_files(".txt")
        jpg_count = self.count_files(".jpg")

        if txt_count == 0 or jpg_count != txt_count:
            print("ERROR:: >>> FileAmount is not correct!")
            return self

        self.images = np.empty((txt_count, self.target_height_resolution, self.target_width_resolution, 3), dtype=np.uint8)
        self.y_class = np.empty(txt_count, dtype=np.int32)
        self.y_bbox = np.empty((txt_count, 4), dtype=np.float32)

        i = 0
        for txt_file in sorted(os.listdir(self.folder)):

            #### CHECK TEXT FILES
            if  txt_file.endswith(".txt"):

                ### IMAGE NAME
                img_name = txt_file.replace(".txt", ".jpg")

                with open(os.path.join(self.folder, txt_file)) as f:
                    parts = f.read().strip().split()


                ### INT ID
                label_id = int(parts[0])

                ### BBOX Array
                bbox = np.array(parts[1:], dtype=np.float32)

                img = Image.open(os.path.join(self.folder, img_name))
                img = img.resize((self.target_width_resolution, self.target_height_resolution))

                if label_id in self.label_names:
                    self.images[i] = np.array(img)
                    self.y_class[i] = label_id
                    self.y_bbox[i] = bbox
                    i += 1
                else:
                    print("label_id is not in Label_list! label_id:",label_id, "img_name:", img_name)

        # Arrays auf tatsächliche Größe kürzen (übersprungene Labels entfernen)
        self.images = self.images[:i]
        self.y_class = self.y_class[:i]
        self.y_bbox = self.y_bbox[:i]

        return self