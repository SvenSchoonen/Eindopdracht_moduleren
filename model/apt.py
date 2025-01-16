import json
import random
import numpy as np

cell_types = {
    "normal": 0,
    "tumor": 1,
    "stem": 2,
    "quiescent": 3,
    "vessel": 4,
    "dead": -1
}

def load_config(file_path="config.json"):
    with open(file_path, "r") as f:
        config = json.load(f)
    return config

class Cell:
    def __init__(self, cell_type, position):
        self.cell_type = cell_type
        self.position = position
        self.alive = True
        self.age = 0
        self.mutation_count = 0
        self.state = 0

    def apoptose(self):
        return random.random() < 0.05

    def proliferate(self):
        return random.random() < 0.3  

    def rest(self):
        pass

    def migrate(self):
        pass

    def mutate(self):
        if random.random() < 0.01:
            self.mutation_count += 1

    def check_neighbors(self):
        pass

def initialize_grid(grid_size, cell_types, normal_ratio=0.9, tumor_ratio=0.1, stem_ratio=0.05):
    grid = np.empty(grid_size, dtype=object)
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            for z in range(grid_size[2]):
                rand_value = random.random()
                if rand_value < normal_ratio:
                    cell_type = cell_types["normal"]
                elif rand_value < tumor_ratio + normal_ratio:
                    cell_type = cell_types["tumor"]
                elif rand_value < stem_ratio + tumor_ratio + normal_ratio:
                    cell_type = cell_types["stem"]
                else:
                    cell_type = cell_types["dead"]

                grid[x, y, z] = Cell(cell_type, (x, y, z))

    return grid

def make_bloodvessel_grid(blood_vessel_place, grid_size, vessel_thickness):
    blood_vessel_grid = np.zeros(grid_size, dtype=int)
    
    x, y = blood_vessel_place
    vessel_x, vessel_y = vessel_thickness
    
    for z in range(grid_size[2]):
        for dx in range(vessel_x):
            for dy in range(vessel_y):
                if (x + dx < grid_size[0]) and (y + dy < grid_size[1]):
                    blood_vessel_grid[x + dx, y + dy, z] = cell_types["vessel"]
    return blood_vessel_grid

def calc_distance_vertical_vessel(grid_size, blood_vessel_grid): # need fixxing
    vessel_positions = np.argwhere(blood_vessel_grid == 4)  # Add config

    if vessel_positions.size == 0:
        raise ValueError("No blood vessels found in the grid!")

    # Create a distance grid initialized with a large value
    min_distances = np.full(grid_size, np.inf)

    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            # Get the Z-coordinates of the vessel positions that match (x, y)
            z_distances = np.abs(vessel_positions[:, 0] - x) + np.abs(vessel_positions[:, 1] - y)
            # Only calculate the minimum distance along the Z-axis
            min_distances[x, y] = np.min(z_distances)
    print(min_distances)
    return min_distances



def create_simulation_grid():
    config = load_config()
    grid = initialize_grid(config["grid_size"], config["cell_types"])
    blood_vessel_grid = make_bloodvessel_grid(config["vessel_place"], config["grid_size"], config["vessel_thickness"])

    for x in range(config["grid_size"][0]):
        for y in range(config["grid_size"][1]):
            for z in range(config["grid_size"][2]):
                if blood_vessel_grid[x, y, z] == cell_types["vessel"]:
                    grid[x, y, z].cell_type = cell_types["vessel"]

    return grid, blood_vessel_grid

grid, blood_vessel_grid = create_simulation_grid()

def simulation_step(grid, blood_vessel_grid):
    config = load_config()
    distance_grid = calc_distance_vertical_vessel(config["grid_size"], blood_vessel_grid)
    print("start stepping")
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            for z in range(grid.shape[2]):
                cell = grid[x, y, z]
                if cell.alive:
                    pass
    return grid

grid = simulation_step(grid, blood_vessel_grid)

def print_grid(grid):
    for z in range(grid.shape[2]):
        result_slice = np.full((grid.shape[0], grid.shape[1]), "empty", dtype=object)
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                cell = grid[x, y, z]
                if cell.cell_type == cell_types["normal"]:
                    result_slice[x, y] = "normal"
                elif cell.cell_type == cell_types["tumor"]:
                    result_slice[x, y] = "tumor"
                elif cell.cell_type == cell_types["stem"]:
                    result_slice[x, y] = "stem"
                elif cell.cell_type == cell_types["dead"]:
                    result_slice[x, y] = "dead"
                elif cell.cell_type == cell_types["vessel"]:
                    result_slice[x, y] = "vessel"
        print(result_slice)
        print()

print_grid(grid)
