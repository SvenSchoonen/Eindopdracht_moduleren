import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import os


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
        self.state = random.choice([0, 1, 2])
        self.mutation_count = 0
        self.mutations = []  # Track mutation history
        self.age = 0

    def apoptose(self):
        config = load_config()
        apoptosis_rate = config["parameters"]["apoptosis_rate"]
        return random.random() < apoptosis_rate

    def proliferate(self, grid, cell_types):
        empty_neighbors = self.check_neighbors(grid, cell_types)

        if empty_neighbors:
            target_pos = random.choice(empty_neighbors)
            grid[target_pos] = Cell(self.cell_type, target_pos)

    def rest(self):
        pass

    def migrate(self, grid):
        neighbors = self.check_neighbors(grid, cell_types)
        empty_neighbors = [pos for pos in neighbors if grid[pos] == cell_types["empty_cell"]]
        if empty_neighbors:
            target_pos = random.choice(empty_neighbors)
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
        cell_types = {
            "normal": 0,
            "tumor": 1,
            "stem": 2,
            "quiescent": 3,
            "vessel": 4,
            "empty_cell": 5,
            "dead": -1
        }

        # List all possible relative positions for neighbors
        neighbors = [
            (dx, dy, dz)
            for dx in range(-1, 2)
            for dy in range(-1, 2)
            for dz in range(-1, 2)
            if not (dx == 0 and dy == 0 and dz == 0)  # Exclude the center cell
        ]

        empty_neighbors = []
        x, y, z = self.position
        for dx, dy, dz in neighbors:
            nx, ny, nz = x + dx, y + dy, z + dz

            # Check if the neighbor position is within the grid bounds
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and 0 <= nz < grid.shape[2]:
                # Get the neighbor cell at this position
                neighbor_cell = grid[nx, ny, nz]
                if neighbor_cell.cell_type == cell_types["empty_cell"]:
                    # Add the position of the empty neighbor to the list
                    empty_neighbors.append((nx, ny, nz))

        return empty_neighbors


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
    config = load_config()
    
    if cell.state == 0:  # Resting state
        cell.rest()
    
    elif cell.state == 1:  # Proliferating state
        if random.random() < config["parameters"]["normal"]["proliferation_rate"]:
            neighbors = cell.check_neighbors(grid, cell_types)            
            empty_neighbors = []
            for pos in neighbors:
                if grid[pos].cell_type == cell_types["empty_cell"]:
                    empty_neighbors.append(pos)
            if empty_neighbors:
                target_pos = random.choice(empty_neighbors)
                grid[target_pos] = Cell(cell_types["normal"], target_pos)  
    
    elif cell.state == 2: 
        cell.migrate(grid)
        
def tumor_cell(cell, grid, cell_types, blood_vessel_grid, distance_grid):
    config = load_config()
    
    # Adjust proliferation rate based on mutation count
    base_proliferation_rate = config["parameters"]["tumor"]["proliferation_rate"]
    mutation_factor = 0.1  # Additional rate per mutation add for in config 
    adjusted_proliferation_rate = base_proliferation_rate + (mutation_factor * cell.mutation_count)
    #print(cell.mutation_count)
    
    if random.random() < adjusted_proliferation_rate:
        neighbors = cell.check_neighbors(grid, cell_types)
        empty_neighbors = [pos for pos in neighbors if grid[pos].cell_type == cell_types["empty_cell"]]
        
        if empty_neighbors:
            target_pos = random.choice(empty_neighbors)
            grid[target_pos] = Cell(cell_types["tumor"], target_pos)  

    if random.random() < config["parameters"]["tumor"]["mutate_rate"]:
        cell.mutate()

    if random.random() < config["parameters"]["tumor"]["migrate_rate"]:
        cell.migrate(grid)
        


def stem_cell(cell, grid, cell_types):
    config = load_config()
    if random.random() < config["parameters"]["stem"]["proliferation_rate"]:
        empty_cells = cell.check_neighbors(grid, cell_types)
        if empty_cells:
            target_pos = random.choice(empty_cells)  # `empty_cells` should be a list of positions
            grid[target_pos[0], target_pos[1], target_pos[2]] = Cell(cell_types["normal"], target_pos)  # Create a normal cell

    if random.random() < config["parameters"]["stem"]["migrate_rate"]:
        cell.migrate(grid)


def vessel_cell(cell, grid, cell_types, distance_grid):
    config = load_config()
    if random.random() < config["vessel_grow"]:
        if cell.cell_type == cell_types["vessel"]:
            x, y, z = cell.position
            if distance_grid[x, y, z] < 2:  
                empty_cells = cell.check_neighbors(grid, cell_types)
                for target_pos in empty_cells:
                    # Check if the target position is adjacent to at least one vessel
                    tx, ty, tz = target_pos
                    neighbors = [
                        (dx, dy, dz)
                        for dx in range(-1, 2)
                        for dy in range(-1, 2)
                        for dz in range(-1, 2)
                        if not (dx == 0 and dy == 0 and dz == 0)
                    ]
                    is_adjacent_to_vessel = False
                    for dx, dy, dz in neighbors:
                        nx, ny, nz = tx + dx, ty + dy, tz + dz
                        if (
                            0 <= nx < grid.shape[0]
                            and 0 <= ny < grid.shape[1]
                            and 0 <= nz < grid.shape[2]
                            and grid[nx, ny, nz].cell_type == cell_types["vessel"]
                        ):
                            is_adjacent_to_vessel = True
                            break
                    if is_adjacent_to_vessel:
                        grid[tx, ty, tz] = Cell(cell_types["vessel"], (tx, ty, tz))
                        break  # Stop after growing the vessel


def add_medicine(grid, blood_vessel_grid, effect_type='cell_death', factor=0.3):
    config = load_config()
    cell_types = config["cell_types"]
    max_age = config["parameters"].get("max_age") 
    
    # Apply effects to all tumor cells
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            for z in range(grid.shape[2]):
                cell = grid[x, y, z]
                if cell.age >= max_age:
                    # Mark the cell as dead when it reaches the max age
                    cell.cell_type = cell_types["dead"]
                elif cell.cell_type == cell_types["tumor"]:
                    if effect_type == 'cell_death':
                        if random.random() < factor:
                            cell.cell_type = cell_types["dead"]
                    elif effect_type == 'slow_growth':
                        config["parameters"]["tumor"]["proliferation_rate"] *= (1 - factor)
                        config["parameters"]["tumor"]["migrate_rate"] *= (1 - factor)

    config["parameters"]["migration_rate"] *= (1 - factor)
    config["parameters"]["mutation_rate"] *= (1 - factor)

    # Save the updated config
    save_config(config)


def clear_dead_cells(grid):
    #    Remove dead cells from the grid.
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            for z in range(grid.shape[2]):
                cell = grid[x, y, z]
                if isinstance(cell, Cell) and not cell.alive:
                    grid[x, y, z] = Cell(cell_types["empty_cell"], (x, y, z))
    return grid

def simulation_step(grid, blood_vessel_grid, distance_grid, step):
    alive_cell_count = 0
    tumor_cell_count = 0
    vessel_cell_count = 0
    normal_count = 0
    clear_dead_steps = 1

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
                        normal_count += 1
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

    if step % clear_dead_steps == 0:
        grid = clear_dead_cells(grid)

    # Calculate total cell count
    config = load_config()
    grid_size = config["grid_size"]
    total_cells = grid_size[0] * grid_size[1] * grid_size[2]

    print(step, "is tep - Alive Cells:", alive_cell_count, "Tumor Cells:", tumor_cell_count,
      "Vessel Cells:", vessel_cell_count, "Normal Cells:", normal_count, "Total Cells:", total_cells)

    return alive_cell_count, tumor_cell_count, vessel_cell_count, normal_count, total_cells

    


def visualize_grid(grid, cell_types):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_vals = []
    y_vals = []
    z_vals = []
    colors = []

    alive_cell_count = 0

    mutation_color_map = plt.get_cmap('cividis')  # Updated to use colormaps 

    cell_type_colors = {
        "normal": 'white',
        "tumor": mutation_color_map(0),  # Default color, will vary based on mutation intensity
        "stem": 'blue',
        "vessel": 'red',
        "empty_cell": 'gray',
        "dead": 'gray'
    }

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
                        colors.append(cell_type_colors["normal"])
                    elif cell.cell_type == cell_types["tumor"]:
                        # Color intensity based on mutation count
                        mutation_intensity = min(cell.mutation_count, 10)  # Limit intensity for visualization
                        color = mutation_color_map(mutation_intensity / 10)  # Normalize by max mutations
                        colors.append(color)
                    elif cell.cell_type == cell_types["stem"]:
                        colors.append(cell_type_colors["stem"])
                    elif cell.cell_type == cell_types["vessel"]:
                        colors.append(cell_type_colors["vessel"])
                    elif cell.cell_type == cell_types["empty_cell"]:
                        colors.append(cell_type_colors["empty_cell"])
                    elif cell.cell_type == cell_types["dead"]:
                        colors.append(cell_type_colors["dead"])

                    alive_cell_count += 1  

    colors = np.array(colors, dtype=object)
    x_vals = np.array(x_vals) 
    y_vals = np.array(y_vals)
    z_vals = np.array(z_vals)

    ax.scatter(x_vals, y_vals, z_vals, c=colors)

    # Create legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cell_type_colors["normal"], markersize=10, label='Norma cell'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cell_type_colors["stem"], markersize=10, label='Stem cell'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cell_type_colors["vessel"], markersize=10, label='Vessel cell'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cell_type_colors["empty_cell"], markersize=10, label='Empty cell'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cell_type_colors["dead"], markersize=10, label='Dead')
    ]

    # For tumor cells, we use a color gradient based on mutation count
    legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=mutation_color_map(0), markersize=10, label='Tumor (Low Mutation)'))
    legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=mutation_color_map(1), markersize=10, label='Tumor (High Mutation)'))

    ax.legend(handles=legend_elements, loc='upper left')

    # Show the plot Cell Properties, Cell Interaction, Cell Interaction, Cell Interaction
    plt.show()

    return fig






def plot_cell_counts(alive_counts, tumor_counts, vessel_counts, normal_counts, total_counts):
    steps = range(len(alive_counts))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(steps, alive_counts, label="Alive Cells", color="blue")
    ax.plot(steps, tumor_counts, label="Tumor Cells", color="black")
    ax.plot(steps, vessel_counts, label="Vessel Cells", color="red")
    ax.plot(steps, normal_counts, label="Normal Cells", color="green")
    ax.plot(steps, total_counts, label="Total Cells", color="red", linestyle="dashed")
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Cell Count")
    ax.set_title("Cell Counts Over Time")
    ax.legend()
    #ax.grid(True)
    return fig
    
def save_plot(fig, step , plot_type="grid"):
    plot_dir = "second_image"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    if plot_type == "grid":
        fig_path = os.path.join(plot_dir, f"grid_step_{step}.png")
    else:
        fig_path = os.path.join(plot_dir, f"plot_step_{step}.png")
    fig.savefig(fig_path)
    print(f"Saved {plot_type} to {fig_path}")
    
    
    
    
# for poster    
alive_counts = []
tumor_counts = []
vessel_counts = []
normal_counts = []
total_counts = []


steps = config["simulation"]["steps"]
grid, blood_vessel_grid = create_simulation_grid()
distance_grid = calc_distance_vertical_vessel(grid.shape, blood_vessel_grid)
for step in range(steps):
    alive, tumor, vessel, normal, total = simulation_step(
        grid, blood_vessel_grid, distance_grid, step
    )
    alive_counts.append(alive)
    tumor_counts.append(tumor)
    vessel_counts.append(vessel)
    normal_counts.append(normal)
    total_counts.append(total)
    #if config["simulation"]["visualize"]:
    #    if step  == 10:
    #        fig_grid = visualize_grid(grid, cell_types) 
    #        save_plot(fig_grid, step, plot_type="grid") 
    if config["simulation"]["medicine"]:        
        if step == 2:
            print("Applying medicine at step 5")
            #sadd_medicine(grid, blood_vessel_grid, effect_type='slow_growth', factor=0.9)
plot_dir = "second_image"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

fig_grid = visualize_grid(grid, cell_types) 
save_plot(fig_grid, step, plot_type="grid") 
plot = plot_cell_counts(alive_counts, tumor_counts, vessel_counts, normal_counts, total_counts)
save_plot(plot, step, plot_type="plot")
