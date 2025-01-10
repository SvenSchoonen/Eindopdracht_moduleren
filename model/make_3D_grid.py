import json
import random 
import numpy as np
def load_config(file_path="config.json"):
    with open(file_path, "r") as f:
        config = json.load(f)
    return config
    

config = load_config()
initialize_grid(config["grid_size"], config["cell_types"])
