import json
# from bunch import Bunch

from utils.parameters import get_parameters

def get_config_from_json(json_file):
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    # config = Bunch(config_dict)

    # return config, config_dict
    return config_dict

def process_config(json_file=""):
    # config, _ = get_config_from_json(json_file)
    config = get_parameters()
    return config