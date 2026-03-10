import os
from helper_classes.config_loader import ConfigLoader

### CONFIG-LOADER
main_folder = os.path.dirname(os.path.abspath(__file__)) + "/"
config = ConfigLoader(main_folder + "config_file.txt")


real_data_train_path = config.train_data_real_path
real_data_test_path = config.test_data_real_path

os.makedirs(real_data_train_path, exist_ok=True)
os.makedirs(real_data_test_path, exist_ok=True)




