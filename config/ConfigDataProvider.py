import configparser


class ConfigDataProvider(object):
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('config/recsys-rl-config.ini')
 
    def get_config_value(self, head_key, key_str):
        return self.config[head_key][key_str]


config_data = ConfigDataProvider()
