"""
this is the config file to pass the constant parameters we use in the project
"""
__author__: str = "Pouya 'Adrian' Firouzmakan"

import json

with open('/Users/pouyafirouzmakan/Desktop/Credit-Card-Fraud-Detection/config/parameters.json', 'r') as f:
    config = json.load(f)
    config['output_path'] = config['output_path'] + config['model']['name'] + ".txt"
    config['dumped_path'] = config['dumped_path'] + config['model']['name'] + ".pkl"

