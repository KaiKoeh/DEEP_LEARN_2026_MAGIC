class ConfigLoader:
    def __init__(self, config_path):
        config = {}
        with open(config_path) as f:
            for line in f.readlines():
                key, value = line.strip().split("=")
                config[key] = int(value)

        self._width = config["IMAGE_WIDTH"]
        self._height = config["IMAGE_HEIGHT"]

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height