import random
import json
import numpy as np
import model  


def load_config(config_file='config.json'):
    """Load the configuration from the given JSON file."""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

class MarkovModel:
    def __init__(self, config_file='config.json'):
        # Load the configuration when initializing the model
        config = load_config(config_file)
        self.transition_probabilities = config['transition_probabilities']
        self.cell_types = config['cell_types']
        self.parameters = config['parameters']
        self.grid_size = config['grid_size']
        
    


    def transition_state(self, current_state):
        probabilities = self.transition_probabilities.get(current_state, {})
        if not probabilities:
            raise ValueError("No transition probabilities defined for state: ", current_state)
        
        next_state = random.choices(
            population=list(probabilities.keys()), 
            weights=list(probabilities.values()), 
            k=1
        )[0]
        
        return next_state



if __name__ == "__main__":
    print("starts")
    model_instance = MarkovModel(config_file='config.json')
    grid, blood_vessel_grid = model.create_simulation_grid() 
    
    distance_grid = model.calc_distance_vertical_vessel(grid.shape, blood_vessel_grid)
    # Initialize the Markov model with the config file parameters
    model_instance = MarkovModel(config_file='config.json')
    
    grid, blood_vessel_grid = model.create_simulation_grid() 
    
    distance_grid = model.calc_distance_vertical_vessel(grid.shape, blood_vessel_grid)
    
    for step in range(10):
        model.simulation_step(grid, blood_vessel_grid, distance_grid, distance_grid, step)
    
    print("Simulation Complete.")
    
