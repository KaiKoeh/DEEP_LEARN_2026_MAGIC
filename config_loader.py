
import os



class ConfigLoader:
    def __init__(self, config_path):
        config = {}
        with open(config_path) as f:
            for line in f.readlines():
                key, value = line.strip().split("=")
                config[key] = int(value)

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
    def img_source_path(self):
        return self._source_path + "img_source/"

    @property
    def model_data_path(self):
        return self._source_path + "model_data/"

    @property
    def model_output_path(self):
        return self._source_path + "model_output/"

    @property
    def label_file_path(self):
        return self._source_path + "model_data/label_file.txt"