import json
import random
import numpy as np

# Cell types
cell_types = {
    "normal": "normal",
    "tumor": "tumor",
    "stem": "stem"
}

# Load the parameters
def load_config(file_path="config.json"):
    with open(file_path, "r") as f:
        config = json.load(f)
    return config

config = load_config()

# Cell class with its behavior
class Cell:
    def __init__(self, cell_type, position):
        self.cell_type = cell_type
        self.position = position  # Position in grid
        self.alive = True
        self.age = 0  # Age or iteration
        self.mutation_count = 0  # Tracks mutations
        self.state = 0  # Default state

    def apoptose(self):
        return False

    def proliferate(self):
        # Proliferate if there is space
        pass 

    def rest(self):
        pass

    def migrate(self):
        # Move cell to a new position (if there is space)
        pass

    def mutate(self):
        pass  # Mutation logic here

    def check_neighbors(self):
        pass  # Check for neighboring cells in the grid


# Initialize the grid with cell types
def initialize_grid(grid_size, cell_types, normal_ratio=0.9, tumor_ratio=0.1, stem_ratio=0.05):
    grid = np.empty(grid_size, dtype=object)

    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            for z in range(grid_size[2]):
                rand_value = random.random()  # Random value to determine cell type
                if rand_value < normal_ratio:
                    cell_type = cell_types["normal"]
                elif rand_value < tumor_ratio + normal_ratio:
                    cell_type = cell_types["tumor"]
                elif rand_value < stem_ratio + tumor_ratio + normal_ratio:
                    cell_type = cell_types["stem"]
                else:
                    continue  # No cell at this position

                grid[x, y, z] = Cell(cell_type, (x, y, z))  # Initialize cell at position

    return grid


# Create blood vessels (simple angiogenesis simulation)
def make_bloodvessel_grid(blood_vessel_place, grid_size, vessel_thickness):
    grid = np.zeros(grid_size, dtype=int)  # Initialize empty grid

    # Coordinates where blood vessels are placed (can be extended or random)
    x, y = blood_vessel_place

    # Simulate simple vertical blood vessel placement along Z axis
    blood_vessel_height = 9
    blood_vessel_thickness_x = vessel_thickness[0]
    blood_vessel_thickness_y = vessel_thickness[1]

    for z in range(blood_vessel_height):
        for dx in range(blood_vessel_thickness_x):
            for dy in range(blood_vessel_thickness_y):
                if x + dx < grid_size[0] and y + dy < grid_size[1] and z < grid_size[2]:
                    grid[x + dx, y + dy, z] = 1  # Mark blood vessel locations

    return grid


# Example grid initialization
config = load_config()

# Initialize the 3D grid
grid = initialize_grid(config["grid_size"], cell_types)

# Example of blood vessel placement and thickness
blood_vessel_grid = make_bloodvessel_grid(config["vessel_place"], config["grid_size"], config["vessel_thickness"])

print("Blood Vessel Grid:")
print(blood_vessel_grid)

