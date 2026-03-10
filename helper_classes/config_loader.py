import os


class ConfigLoader:
    def __init__(self, config_path):
        config = {}

        with open(config_path) as f:
            for line in f.readlines():
                line = line.strip()

                if not line or "=" not in line:
                    continue

                key, value = line.split("=", 1)

                try:
                    config[key] = int(value)
                except ValueError:
                    config[key] = value

        ### LOCAL CONFIGS
        self._width = config["IMAGE_WIDTH"]
        self._height = config["IMAGE_HEIGHT"]
        self._data_source = config["DATA_SOURCE"]

        ### SOURCE PATHS
        self._project_path = os.path.dirname(os.path.abspath(config_path)) + "/"
        self._source_path = self._project_path + "data_source/" + self._data_source + "/"

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def data_source(self):
        return self._data_source

    @property
    def project_path(self):
        return self._project_path

    # --- img_source Pfade ---
    @property
    def img_source_path(self):
        return self._source_path + "img_source/"

    @property
    def photo_raw_path(self):
        return self._source_path + "img_source/0_photo_raw/"

    @property
    def photo_raw_skip_path(self):
        return self._source_path + "img_source/0_photo_raw_skip/"

    @property
    def photos_sorted_path(self):
        return self._source_path + "img_source/1_photos_sorted/"

    @property
    def photos_finished_path(self):
        return self._source_path + "img_source/2_photos_finished/"

    @property
    def scryfall_cards_path(self):
        return self._source_path + "img_source/_scryfall_cards/"

    # --- model_data Pfade ---
    @property
    def model_data_path(self):
        return self._source_path + "model_data/"

    @property
    def label_file_path(self):
        return self._source_path + "model_data/label_file.txt"

    @property
    def train_data_synthetic_path(self):
        return self._source_path + "model_data/train_data_synthetic/"

    @property
    def test_data_synthetic_path(self):
        return self._source_path + "model_data/test_data_synthetic/"

    @property
    def test_data_real_path(self):
        return self._source_path + "model_data/test_data_real/"

    @property
    def train_data_real_path(self):
        return self._source_path + "model_data/train_data_real/"

    # --- model_output Pfade ---
    @property
    def model_output_path(self):
        return self._source_path + "model_output/"
