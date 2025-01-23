import json
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the cell types
cell_types = {
    "normal": 0,
    "tumor": 1,
    "stem": 2,
    "quiescent": 3,
    "vessel": 4,
    "empty_cell": 5,
    "dead": -1
}

def load_config(file_path="config.json"):
    """ Load configuration from a JSON file. """
    with open(file_path, "r") as f:
        config = json.load(f)
    return config

class Cell:
    """ Represents a single cell in the grid. """
    def __init__(self, cell_type, position):
        self.cell_type = cell_type
        self.position = position
        self.alive = True
        self.age = 0
        self.mutation_count = 0
        self.state = random.randint(0, 3)  # Random state (0 = resting, 1 = growing, 2 = dividing)

    def apoptose(self):
        """ Apoptosis (cell death) based on a random chance. """
        return random.random() < 0.05

    def proliferate(self):
        """ Proliferation chance. """
        return random.random() < 0.3

    def rest(self):
        """ Resting state. """
        pass

    def migrate(self):
        """ Migration of the cell. """
        pass

    def mutate(self):
        """ Mutation event. """
        if random.random() < 0.01:
            self.mutation_count += 1  # Increment mutation count

    def check_neighbors(self, grid, cell_types):
        """ Check neighbors to find empty cells for proliferation. """
        empty_cells_interest = {}
        neighbors = [
            (dx, dy, dz)
            for dx in range(-1, 2)
            for dy in range(-1, 2)
            for dz in range(-1, 2)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]
        x, y, z = self.position
        for dx, dy, dz in neighbors:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and 0 <= nz < grid.shape[2]:
                neighbor_cell = grid[nx, ny, nz]
                if neighbor_cell.cell_type == cell_types["empty_cell"]:
                    if (nx, ny, nz) not in empty_cells_interest:
                        empty_cells_interest[(nx, ny, nz)] = []
                    empty_cells_interest[(nx, ny, nz)].append((x, y, z))
        return empty_cells_interest

def initialize_grid(grid_size, cell_types, normal_ratio=0.5, tumor_ratio=0.1, stem_ratio=0.05, empty_cell_ratio=0.4):
    """ Initialize the grid with cells based on the given ratios. """
    grid = np.empty(grid_size, dtype=object)
    total_probability = normal_ratio + tumor_ratio + stem_ratio + empty_cell_ratio
    if total_probability < 1:
        print("Total probability of cells does not add up to 1.")
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
                elif rand_value < empty_cell_ratio + stem_ratio + tumor_ratio + normal_ratio:
                    cell_type = cell_types["empty_cell"]
                else:
                    cell_type = cell_types["dead"]
                grid[x, y, z] = Cell(cell_type, (x, y, z))
    return grid

def make_bloodvessel_grid(blood_vessel_place, grid_size, vessel_thickness):
    """ Create a grid with blood vessels. """
    blood_vessel_grid = np.zeros(grid_size, dtype=int)
    x, y = blood_vessel_place
    vessel_x, vessel_y = vessel_thickness
    for z in range(grid_size[2]):
        for dx in range(vessel_x):
            for dy in range(vessel_y):
                if (x + dx < grid_size[0]) and (y + dy < grid_size[1]):
                    blood_vessel_grid[x + dx, y + dy, z] = cell_types["vessel"]
    return blood_vessel_grid

def calc_distance_vertical_vessel(grid_size, blood_vessel_grid):
    """ Calculate the distance to the nearest vertical blood vessel. """
    vessel_positions = np.argwhere(blood_vessel_grid == cell_types["vessel"])
    if vessel_positions.size == 0:
        raise ValueError("No blood vessels found in the grid!")
    distance_grid = np.full(grid_size, np.inf)
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            for z in range(grid_size[2]):
                distances = np.sqrt(((vessel_positions - [x, y, z]) ** 2).sum(axis=1))
                distance_grid[x, y, z] = np.min(distances)
    return distance_grid

def create_simulation_grid():
    """ Create the simulation grid with cells and blood vessels. """
    config = load_config()
    grid = initialize_grid(config["grid_size"], config["cell_types"])
    blood_vessel_grid = make_bloodvessel_grid(config["vessel_place"], config["grid_size"], config["vessel_thickness"])

    for x in range(config["grid_size"][0]):
        for y in range(config["grid_size"][1]):
            for z in range(config["grid_size"][2]):
                if blood_vessel_grid[x, y, z] == cell_types["vessel"]:
                    grid[x, y, z].cell_type = cell_types["vessel"]

    return grid, blood_vessel_grid

def normal_cell(cell, grid, cell_types):
    """ Logic for normal cells. """
    if cell.state == 0:  # Resting phase
        cell.rest()
    
    elif cell.state == 1:  # Growing state
        empty_cells = cell.check_neighbors(grid, cell_types)
        if empty_cells:
            target_pos = random.choice(list(empty_cells.keys()))
            if random.random() < 0.3:  # 30% chance to proliferate
                grid[target_pos[0], target_pos[1], target_pos[2]] = Cell(cell_types["normal"], target_pos)
        
        # Mutation possibility
        cell.mutate()
    
    elif cell.state == 2:  # Dividing state
        cell.migrate()

def tumor_cell(cell, grid, cell_types, blood_vessel_grid, distance_grid):
    """ Logic for tumor cells. """
    if random.random() < 0.4:  # Tumor cells are more likely to proliferate (40%)
        empty_cells = cell.check_neighbors(grid, cell_types)
        if empty_cells:
            target_pos = random.choice(list(empty_cells.keys()))
            grid[target_pos[0], target_pos[1], target_pos[2]] = Cell(cell_types["tumor"], target_pos)
    
    if random.random() < 0.2:  # 20% chance to mutate and become more aggressive
        cell.mutation_count += 1
    
    x, y, z = cell.position
    if distance_grid[x, y, z] < 2:  # Tumor cells near blood vessels (distance < 2)
        if random.random() < 0.9:  # 90% chance to proliferate near a blood vessel
            empty_cells = cell.check_neighbors(grid, cell_types)
            if empty_cells:
                target_pos = random.choice(list(empty_cells.keys()))
                grid[target_pos[0], target_pos[1], target_pos[2]] = Cell(cell_types["tumor"], target_pos)

def stem_cell(cell, grid, cell_types):
    """ Logic for stem cells. """
    if random.random() < 0.5:  # 50% chance to proliferate (divide)
        empty_cells = cell.check_neighbors(grid, cell_types)
        if empty_cells:
            target_pos = random.choice(list(empty_cells.keys()))
            grid[target_pos[0], target_pos[1], target_pos[2]] = Cell(cell_types["stem"], target_pos)
    
    if random.random() < 0.3:  # 30% chance to migrate
        cell.migrate()

def simulation_step(grid, blood_vessel_grid, cell_types, distance_grid):
    """ Run a simulation step and update cell states. """
    alive_cell_count = 0  # To track the number of alive cells during this step
    tumor_cell_count = 0  # To track the number of tumor cells during this step
    
    for z in range(grid.shape[2]):
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                cell = grid[x, y, z]
                if cell.alive:
                    cell.age += 1  # Increase the cell's age
                    
                    if cell.cell_type == cell_types["normal"]:
                        normal_cell(cell, grid, cell_types)
                    elif cell.cell_type == cell_types["tumor"]:
                        tumor_cell(cell, grid, cell_types, blood_vessel_grid, distance_grid)
                    elif cell.cell_type == cell_types["stem"]:
                        stem_cell(cell, grid, cell_types)
                    
                    if cell.apoptose():  # Check for apoptosis (death)
                        cell.alive = False
                        grid[x, y, z] = Cell(cell_types["dead"], (x, y, z))
                    
                    if cell.cell_type == cell_types["tumor"]:
                        tumor_cell_count += 1
                    if cell.alive:
                        alive_cell_count += 1

    return alive_cell_count, tumor_cell_count

def visualize_grid(grid, cell_types):
    """ Visualize the grid in 3D. """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            for z in range(grid.shape[2]):
                cell = grid[x, y, z]
                if cell.alive:
                    ax.scatter(x, y, z, c='g' if cell.cell_type == cell_types["normal"] else 'r' if cell.cell_type == cell_types["tumor"] else 'b', marker='o')
    
    plt.show()

# Running the simulation
grid_size = (20, 20, 20)
config = load_config()
grid, blood_vessel_grid = create_simulation_grid()
distance_grid = calc_distance_vertical_vessel(grid_size, blood_vessel_grid)

# Simulation steps
for step in range(10):  # Run for 100 steps
    alive_cell_count, tumor_cell_count = simulation_step(grid, blood_vessel_grid, config["cell_types"], distance_grid)
    if step % 2 == 0:  # Visualize every 10 steps
        visualize_grid(grid, config["cell_types"])

    print(f"Step {step}: Alive cells = {alive_cell_count}, Tumor cells = {tumor_cell_count}")
