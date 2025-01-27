import random
import json
import numpy as np


def load_config(config_file='config.json'):
    """Load the configuration from the given JSON file."""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


class MarkovModel:
    def __init__(self, config_file='config.json'):
        # Load the configuration when initializing the model
        config = load_config(config_file)
        self.cell_types = config['cell_types']
        self.parameters = config['parameters']
        self.grid_size = config['grid_size']
        
        # Dynamically calculate transition probabilities
        self.transition_probabilities = self.calculate_transition_probabilities()

    def calculate_transition_probabilities(self):
        """Calculate and normalize transition probabilities for each state."""
        transition_probabilities = {}
        for state, params in self.parameters.items():
            if state not in self.cell_types:
                continue 

            proliferation_rate = params.get("proliferation_rate", 0)
            max_division_rate = params.get("max_division_rate", 0)
            migration_rate = self.parameters.get("migration_rate", 0.05)

            # Calculate base probabilities
            transition_probabilities[state] = {
                "normal": proliferation_rate * 0.5,
                "tumor": max_division_rate * 0.3,
                "stem": migration_rate * 0.2,
                "vessel": migration_rate * 0.1,
                "empty_cell": migration_rate * 0.4,  
            }

            # Normalize probabilities
            total = sum(transition_probabilities[state].values())
            transition_probabilities[state] = {
                k: v / total for k, v in transition_probabilities[state].items()
            }

        # Special handling for 'vessel'
        transition_probabilities["vessel"] = {
            "vessel": 1.0  # Vessel remains unchanged
        }

        return transition_probabilities

    def transition_state(self, current_state):
        """Perform a state transition based on current state probabilities."""
        probabilities = self.transition_probabilities.get(current_state, {})

        if not probabilities:
            return current_state  # No transition defined, stay in current state

        next_state = random.choices(
            population=list(probabilities.keys()), 
            weights=list(probabilities.values()), 
            k=1
        )[0]
        return next_state

    def predict_grid(self, grid):
        """Predict the next states for the entire grid."""
        new_grid = np.copy(grid)  # Create a copy to store the new states

        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                for z in range(grid.shape[2]):
                    current_state = grid[x, y, z]
                    # Predict the next state using the Markov model
                    next_state = self.transition_state(current_state)
                    new_grid[x, y, z] = next_state

        return new_grid
