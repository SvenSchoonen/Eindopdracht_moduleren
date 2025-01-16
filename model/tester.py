import json
import random
import numpy as np

# Cell types mapping from configuration for 
cell_types = {
    "normal": 0,
    "tumor": 1,
    "stem": 2,
    "quiescent": 3,
    "vessel": 4,
    "dead": -1
}

# Load the configuration
def load_config(file_path="config.json"):
    with open(file_path, "r") as f:
        config = json.load(f)
    return config

# Cell class 
class Cell:
    def __init__(self, cell_type, position):
        self.cell_type = cell_type
        self.position = position  # Position in grid
        self.alive = True
        self.age = 0  # Age or iteration
        self.mutation_count = 0  # Tracks mutations
        self.state = 0  # Default state

    def apoptose(self):
        # Apoptosis behavior
        return random.random() < 0.05  # 5 procent change

    def proliferate(self):
        # Proliferate if there is space
        return random.random() < 0.3  

    def rest(self):
        pass

    def migrate(self):
        # Move cell to a new position (if there is space)
        pass

    def mutate(self):
        if random.random() < 0.01:  # Example mutation rate of 1%
            self.mutation_count += 1 # first type of mutations 

    def check_neighbors(self):
        pass  # Check for neighboring cells in the grid


# Initialize the grid with cell types
def initialize_grid(grid_size, cell_types, normal_ratio=0.9, tumor_ratio=0.1, stem_ratio=0.05):
    grid = np.empty(grid_size, dtype=object)
    # Use np.mgrid
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
                    cell_type = cell_types["dead"]  # For now, dead or empty cells are assigned default

                grid[x, y, z] = Cell(cell_type, (x, y, z))  # Initialize cell at position

    return grid

# Blood vessel grid (placement and thickness)
def make_bloodvessel_grid(blood_vessel_place, grid_size, vessel_thickness):
    blood_vessel_grid = np.zeros(grid_size, dtype=int)
    
    x, y = blood_vessel_place
    vessel_x, vessel_y = vessel_thickness
    
    # Place the blood vessel in the grid
    for z in range(grid_size[2]):  # Assuming the blood vessel extends along z-axis
        for dx in range(vessel_x):
            for dy in range(vessel_y):
                if (x + dx < grid_size[0]) and (y + dy < grid_size[1]):
                    blood_vessel_grid[x + dx, y + dy, z] = cell_types["vessel"]  # Mark blood vessel
    return blood_vessel_grid
    
def calc_distance_vessel(grid_size, blood_vessel_grid):    
    # Create a 3D grid of positions using np.mgrid
    x, y, z = np.mgrid[:grid_size[0], :grid_size[1], :grid_size[2]]

    positions = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T  # Shape (N, 3)

    vessel_positions = np.argwhere(blood_vessel_grid == 1)  # Shape (M, 3), M = number of blood vessels
    
    # Compute the Chebyshev distance from all positions to all blood vessel positions
    distances = np.max(np.abs(positions[:, np.newaxis, :] - vessel_positions), axis=2)
    print("distance :", distances)
    distance_grid = distances.min(axis=1).reshape(grid_size)
    
    return distance_grid


def create_simulation_grid():
    config = load_config()  
    # make cell grid
    grid = initialize_grid(config["grid_size"], config["cell_types"])
    # Make vessel grid
    blood_vessel_grid = make_bloodvessel_grid(config["vessel_place"], config["grid_size"], config["vessel_thickness"])
    
    # Combine the grids (Blood vessels and Cells)
    for x in range(config["grid_size"][0]):
        for y in range(config["grid_size"][1]):
            for z in range(config["grid_size"][2]):
                if blood_vessel_grid[x, y, z] == cell_types["vessel"]:
                    grid[x, y, z].cell_type = cell_types["vessel"]  # Mark as vessel in cell grid

    return grid, blood_vessel_grid


grid, blood_vessel_grid = create_simulation_grid()

# Simulation step: Each cell may perform actions.
def simulation_step(grid, blood_vessel_grid):
    config = load_config()  
    calc_distance_vessel(config["grid_size"], blood_vessel_grid)
    print("start stepping")
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            for z in range(grid.shape[2]):
                cell = grid[x, y, z]
                if cell.alive:
                    print("start cell action")  


grid = simulation_step(grid, blood_vessel_grid) 
# SHOW GRID 
#print("Resulting Grid with Blood Vessels and Cells:")
#for z in range(grid.shape[2]):
#    print(f"z={z} slice:")
#    result_slice = np.full((grid.shape[0], grid.shape[1]), "empty", dtype=object)
#    for x in range(grid.shape[0]):
#        for y in range(grid.shape[1]):
#            cell = grid[x, y, z]
#            if cell.cell_type == cell_types["normal"]:
#                result_slice[x, y] = "normal"
#            elif cell.cell_type == cell_types["tumor"]:
#               result_slice[x, y] = "tumor"
#            elif cell.cell_type == cell_types["stem"]:
#                result_slice[x, y] = "stem"
#            elif cell.cell_type == cell_types["dead"]:
#                result_slice[x, y] = "dead"
#            elif cell.cell_type == cell_types["vessel"]:
#                result_slice[x, y] = "vessel"
#    print(result_slice)
#    print()
