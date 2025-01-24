import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


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
    with open(file_path, "r") as f:
        config = json.load(f)
    return config

config = load_config()

class Cell:
    def __init__(self, cell_type, position):
        self.cell_type = cell_type
        self.position = position
        self.alive = True
        self.state = 0  # 0: resting, 1: growing, 2: dividing
        self.mutation_count = 0
        self.mutations = []  # Track mutation history
        self.age = 0

    def apoptose(self):
        config = load_config()
        apoptosis_rate = config["parameters"]["apoptosis_rate"] 
        return random.random() < apoptosis_rate

    def proliferate(self):
        config = load_config()
        proliferation_rate = config["parameters"]["normal"]["proliferation_rate"]
        return random.random() < proliferation_rate 

    def rest(self):
        pass

    def migrate(self, grid):
        neighbors = self.check_neighbors(grid, cell_types)
        empty_neighbors = [pos for pos in neighbors if grid[pos] == cell_types["empty_cell"]]
        if empty_neighbors:
            target_pos = random.choice(empty_neighbors)
            # Update the grid and the cell's position
            grid[target_pos] = Cell(self.cell_type, target_pos)
            grid[self.position] = cell_types["empty_cell"]
            self.position = target_pos
    
    def mutate(self):
        mutation_type = random.choice(["aggressive", "less_aggressive"])
        if mutation_type == "aggressive":
            self.mutation_count += 1
        elif mutation_type == "less_aggressive" and self.mutation_count > 0:
            self.mutation_count -= 1
        self.mutations.append(mutation_type)

    def check_neighbors(self, grid, cell_types):
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


def initialize_grid(grid_size, cell_types, normal_ratio=0.5, tumor_ratio=0.1, stem_ratio=0.01, empty_cell_ratio=0.4):
    grid = np.empty(grid_size, dtype=object)
    total_probability = normal_ratio + tumor_ratio + stem_ratio + empty_cell_ratio
    if total_probability < 1:
        print("Total_probability of cells does not add up to 1.")

    positions = np.mgrid[0:grid_size[0], 0:grid_size[1], 0:grid_size[2]].reshape(3, -1).T
    for x, y, z in positions:
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
    blood_vessel_grid = np.zeros(grid_size, dtype=int)
    x, y = blood_vessel_place
    vessel_x, vessel_y = vessel_thickness
    for z in range(grid_size[2]):# always max length
        for dx in range(vessel_x):
            for dy in range(vessel_y):
                if (x + dx < grid_size[0]) and (y + dy < grid_size[1]):
                    blood_vessel_grid[x + dx, y + dy, z] = cell_types["vessel"]
    return blood_vessel_grid

def calc_distance_vertical_vessel(grid_size, blood_vessel_grid):
    vessel_positions = np.argwhere(blood_vessel_grid == cell_types["vessel"])
    if vessel_positions.size == 0:
        raise ValueError("No blood vessels found in the grid!")

    grid_positions = np.mgrid[0:grid_size[0], 0:grid_size[1], 0:grid_size[2]].reshape(3, -1).T
    distance_grid = np.full(grid_size, np.inf)

    for pos in grid_positions:
        distances = np.sqrt(((vessel_positions - pos) ** 2).sum(axis=1))
        distance_grid[tuple(pos)] = np.min(distances)
    return distance_grid

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

def normal_cell(cell, grid, cell_types):
    if cell.state == 0:  # Resting phase
        cell.rest()
    elif cell.state == 1:  # Growing state
        if random.random() < 0.3:  # 30% chance to proliferate
            cell.migrate(grid)
        cell.mutate()
    elif cell.state == 2:  # Dividing state
        cell.migrate(grid)
   
    elif cell.state == 2:  
        cell.migrate(grid)

def tumor_cell(cell, grid, cell_types, blood_vessel_grid, distance_grid):
    if random.random() < 0.4:  # Tumor cells are more likely to proliferate (40%)
        cell.migrate(grid)
    if random.random() < 0.2:  # 20% chance to mutate
        cell.mutate()
    x, y, z = cell.position
    if distance_grid[x, y, z] < 2:  # Tumor cells near blood vessels (distance < 2)
        if random.random() < 0.9:  # 90% chance to proliferate near a blood vessel
            cell.migrate(grid)


def stem_cell(cell, grid, cell_types):
    if random.random() < 0.5:  # 50% chance to proliferate (divide)
        empty_cells = cell.check_neighbors(grid, cell_types)
        if empty_cells:
            target_pos = random.choice(list(empty_cells.keys()))
            grid[target_pos[0], target_pos[1], target_pos[2]] = Cell(cell_types["normal"], target_pos) # makes a normal cell
   
    if random.random() < 0.3:  # 30% chance to migrate
        cell.migrate(grid)

def vessel_cell(cell, grid, cell_types, distance_grid):
    if cell.cell_type == cell_types["vessel"]:
        x, y, z = cell.position
        if distance_grid[x, y, z] < 2:  # Blood vessels grow toward tumor cells
            empty_cells = cell.check_neighbors(grid, cell_types)
            if empty_cells:
                target_pos = random.choice(list(empty_cells.keys()))
                grid[target_pos[0], target_pos[1], target_pos[2]] = Cell(cell_types["vessel"], target_pos)

def add_medicine(grid, blood_vessel_grid, effect_type, factor):
    if effect_type == 'vessel_growth':
        # Reduce vessel growth rate
        return grid * (1 - factor)  # Example: decreases proliferation by factor
    elif effect_type == 'cell_death':
        # Increase cell death rate
        mask = np.random.rand(*grid.shape) < factor
        grid[mask] = 0  # Cells die in the affected regions
    elif effect_type == 'resistance':
        # Increase resistance probability
        resistance_chance = np.random.rand(*grid.shape)
        grid[resistance_chance < factor] += 1  # Resistance markers increase
    return grid


def simulation_step(grid, blood_vessel_grid, distance_grid, step):
    alive_cell_count = 0
    tumor_cell_count = 0
    vessel_cell_count = 0
    cell_types = {
    "normal": 0,
    "tumor": 1,
    "stem": 2,
    "quiescent": 3,
    "vessel": 4,
    "empty_cell": 5,
    "dead": -1
    }
    for z in range(grid.shape[2]):
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                cell = grid[x, y, z]
                if cell.alive:
                    cell.age += 1
                    if cell.cell_type == cell_types["normal"]:
                        normal_cell(cell, grid, cell_types)
                    elif cell.cell_type == cell_types["tumor"]:
                        tumor_cell(cell, grid, cell_types, blood_vessel_grid, distance_grid)
                        tumor_cell_count += 1
                    elif cell.cell_type == cell_types["stem"]:
                        stem_cell(cell, grid, cell_types)
                    elif cell.cell_type == cell_types["vessel"]:
                        vessel_cell(cell, grid, cell_types, distance_grid)
                        vessel_cell_count += 1
                   
                    if cell.apoptose():
                        cell.alive = False
                    else:
                        alive_cell_count += 1
    config = load_config()
    grid_size = config["grid_size"]
    totaal =  grid_size[0] * grid_size[1] * grid_size[2]
    print("Alive Cells:", alive_cell_count, "Tumor Cells: " ,tumor_cell_count, "Vessel Cells: ", vessel_cell_count, "Totaal cells: " , totaal)
    return alive_cell_count, tumor_cell_count, vessel_cell_count



def visualize_blood_vessel_and_mutations(grid, step):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    positions = np.mgrid[0:grid.shape[0], 0:grid.shape[1], 0:grid.shape[2]].reshape(3, -1).T
    x_vals, y_vals, z_vals, colors, sizes = [], [], [], [], []

    for x, y, z in positions:
        cell = grid[x, y, z]
        if isinstance(cell, Cell) and cell.alive:
            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(z)

            if cell.cell_type == cell_types["vessel"]:
                colors.append("red")  # Blood vessels
            elif cell.cell_type == cell_types["tumor"]:
                mutation_intensity = min(cell.mutation_count, 10)  # Limit size growth
                sizes.append(50 + mutation_intensity * 10)
                colors.append("black")
            else:
                colors.append("white")  # Normal cells
                sizes.append(50)

    ax.scatter(x_vals, y_vals, z_vals, c=colors, s=sizes, marker="o")
    ax.set_title(f"Step {step}: Tumor Aggression and Vessels")
    plt.show()

def visualize_grid(grid): 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_vals = []
    y_vals = []
    z_vals = []
    colors = []

    alive_cell_count = 0 

    for z in range(grid.shape[2]):
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                cell = grid[x, y, z]
               
                if cell.alive:  # Only add alive cells to the plot
                    x_vals.append(x)
                    y_vals.append(y)
                    z_vals.append(z)

                    # Assign color based on cell type
                    if cell.cell_type == cell_types["normal"]:
                        colors.append('white')
                    elif cell.cell_type == cell_types["tumor"]:
                        colors.append('black')
                    elif cell.cell_type == cell_types["stem"]:
                        colors.append('blue')
                    elif cell.cell_type == cell_types["vessel"]:
                        colors.append('red')
                    elif cell.cell_type == cell_types["empty_cell"]:
                        colors.append('gray')  # Optional color for empty cells
                    elif cell.cell_type == cell_types["dead"]:
                        colors.append('gray')  # Optional color for dead cells

                    alive_cell_count += 1  # Increment alive cell count

    else:
        # Plot the cells in 3D
        ax.scatter(x_vals, y_vals, z_vals, c=colors)

    # Show the plot
    plt.show(),
step = config["simulation"]["steps"]
grid, blood_vessel_grid = create_simulation_grid()
distance_grid = calc_distance_vertical_vessel(grid.shape, blood_vessel_grid)

for step in range(step):  # Run simulation for 10 steps
    print("step", step)
    
    simulation_step(grid, blood_vessel_grid, distance_grid=distance_grid, step=step)
    config = load_config() 
    if config["simulation"]["visualize"] == True:
        
        visualize_grid(grid)
    if step == 5:  # Apply medicine at step 5 still need fixing 
        add_medicine(grid, blood_vessel_grid, effect_type='cell_death', factor=0.3)


