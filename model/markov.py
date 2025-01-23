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
        self.cell_types = config['cell_types']
        self.parameters = config['parameters']
        self.grid_size = config['grid_size']
        
        # Dynamically calculate transition probabilities
        self.transition_probabilities = self.calculate_transition_probabilities()

    def calculate_transition_probabilities(self):
        """
        Calculate transition probabilities dynamically based on the parameters in the configuration.
        Includes support for 'empty_cell' state.
        """
        transition_probabilities = {}
        for state, params in self.parameters.items():
            if state not in self.cell_types:
                continue  # Skip invalid states

            # Define transition weights using state parameters
            proliferation_rate = params.get("proliferation_rate", 0)
            max_division_rate = params.get("max_division_rate", 0)
            migration_rate = self.parameters.get("migration_rate", 0.05)

            # Example logic for calculating transition probabilities
            transition_probabilities[state] = {
                "normal": proliferation_rate * 0.5,
                "tumor": max_division_rate * 0.3,
                "stem": migration_rate * 0.2,
                "vessel": migration_rate * 0.1,
                "empty_cell": migration_rate * 0.4,  
            }

            # Normalize probabilities to ensure they sum up to 1
            total = sum(transition_probabilities[state].values())
            transition_probabilities[state] = {
                k: v / total for k, v in transition_probabilities[state].items()
            }

        # Special handling for 'empty_cell'
        transition_probabilities["vessel"] = {
            "vessel": 4.0 # vessel remains the same
        }

        return transition_probabilities

    def transition_state(self, current_state):
        """Transition from the current state to the next state based on probabilities."""
        probabilities = self.transition_probabilities.get(current_state, {})

        next_state = random.choices(
            population=list(probabilities.keys()), 
            weights=list(probabilities.values()), 
            k=1
        )[0]
        return next_state


if __name__ == "__main__":
    print("Simulation starts...")

    # Load and initialize the Markov model
    model_instance = MarkovModel(config_file='config.json')

    # Create simulation grid and calculate the distance grid
    grid, blood_vessel_grid = model.create_simulation_grid()
    distance_grid = model.calc_distance_vertical_vessel(grid.shape, blood_vessel_grid)

    # Run the simulation for defined steps
    for step in range(10):
        print(step + 1)

        # Simulate a single grid cell's state change for demonstration
        current_state = "empty_cell"  # Example current state
        next_state = model_instance.transition_state(current_state)
        print(f"Current state: {current_state} -> Next state: {next_state}")

 
