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

class Cell:
    def __init__(self, cell_type, position):
        self.cell_type = cell_type
        self.position = position
        self.alive = True
        self.age = 0
        self.mutation_count = 0
        self.state = random.randint(0, 3)  # random state (0 = rest, 1 = growing, 2 = dividing)

    def apoptose(self):
        return random.random() < 0.05

    def proliferate(self):
        return random.random() < 0.3

    def rest(self):
        pass

    def migrate(self, grid):
        neighbors = self.check_neighbors(grid, cell_types)
        empty_neighbors = [pos for pos in neighbors if grid[pos] == cell_types["empty_cell"]]
        if empty_neighbors:
            new_position = random.choice(empty_neighbors)
            grid[new_position] = self.cell_type
            grid[self.position] = cell_types["empty_cell"]
            self.position = new_position

    def mutate(self):
        if random.random() < 0.01:
            self.mutation_count += 1  # Increment mutation count

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

def initialize_grid(grid_size, cell_types, normal_ratio=0.5, tumor_ratio=0.1, stem_ratio=0.05, empty_cell_ratio=0.4):
    grid = np.empty(grid_size, dtype=object)
    total_probability = normal_ratio + tumor_ratio + stem_ratio + empty_cell_ratio
    if total_probability < 1:
        print("Total_probability of cells does not add up to 1.")
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
        empty_cells = cell.check_neighbors(grid, cell_types)
        if empty_cells:
            target_pos = random.choice(list(empty_cells.keys()))
            if random.random() < 0.3:  # 30% chance to proliferate
                grid[target_pos[0], target_pos[1], target_pos[2]] = Cell(cell_types["normal"], target_pos)
       
        # Mutation possibility
        cell.mutate()
   
    elif cell.state == 2:  
        cell.migrate(grid)

def tumor_cell(cell, grid, cell_types, blood_vessel_grid, distance_grid):
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
    if random.random() < 0.5:  # 50% chance to proliferate (divide)
        empty_cells = cell.check_neighbors(grid, cell_types)
        if empty_cells:
            target_pos = random.choice(list(empty_cells.keys()))
            grid[target_pos[0], target_pos[1], target_pos[2]] = Cell(cell_types["stem"], target_pos)
   
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

# Data collection for prediction
data = []
labels = []

def collect_data(grid, step):
    step_data = []
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            for z in range(grid.shape[2]):
                cell = grid[x, y, z]
                cell_features = [
                    x, y, z,  # Position
                    cell.cell_type,  # Type of cell
                    cell.mutation_count,  # Mutations
                    cell.state,  # State
                ]
                step_data.append(cell_features)
                labels.append(step + 1)  # Label the next step
    data.extend(step_data)

def simulation_step(grid, blood_vessel_grid, cell_types, distance_grid, step):
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
    collect_data(grid, step)  # Collect data for prediction
    grid_size = config["grid_size"]
    totaal =  grid_size[0] * grid_size[1] * grid_size[2]
    print("Alive Cells:", alive_cell_count, "Tumor Cells: " ,tumor_cell_count, "Vessel Cells: ", vessel_cell_count, "Totaal cells: " , totaal)
    return alive_cell_count, tumor_cell_count, vessel_cell_count



def visualize_blood_vessel_and_mutations(grid, blood_vessel_grid, cell_types, distance_grid, step):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_vals = []
    y_vals = []
    z_vals = []
    colors = []

    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            for z in range(grid.shape[2]):
                cell = grid[x, y, z]
                if cell.alive:
                    x_vals.append(x)
                    y_vals.append(y)
                    z_vals.append(z)
                    if cell.cell_type == cell_types["vessel"]:
                        color = (1, 0, 0)  # red for vessel
                    else:
                        color = (1, 1, 1)  # grey for others
                    colors.append(color)
                # Show mutations by increasing the size of mutated tumor cells
                if cell.cell_type == cell_types["tumor"] and cell.mutation_count > 0:
                    ax.scatter(x, y, z, c='black', marker='o', s=50 + cell.mutation_count * 10)
    
    ax.scatter(x_vals, y_vals, z_vals, c=colors, marker='o')
    ax.set_title(step,": Blood Vessels and Tumor Mutations")
    plt.show()



def train_knn_predictive_model(grid):
    data = []
    labels = []

    # Collecting features from the grid (position, cell_type, mutation_count, and state)
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            for z in range(grid.shape[2]):
                cell = grid[x, y, z]
                cell_features = [
                    x, y, z,  # Position
                    cell.cell_type,  # Type of cell
                    cell.mutation_count,  # Mutations
                    cell.state,  # State
                ]
                data.append(cell_features)
                labels.append(cell.state)  # Predicting the next state
    
    data = np.array(data)
    labels = np.array(labels)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Train KNN model
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)
    
    accuracy = knn_model.score(X_test, y_test)
    print(f"KNN Model accuracy: {accuracy}")
    return knn_model

# KNN prediction
def predict_next_step_knn(grid, model):
    step_data = []
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            for z in range(grid.shape[2]):
                cell = grid[x, y, z]
                cell_features = [
                    x, y, z,  # Position
                    cell.cell_type,  # Type of cell
                    cell.mutation_count,  # Mutations
                    cell.state,  # State
                ]
                step_data.append(cell_features)
    
    predictions = model.predict(step_data)

    # Update grid with predictions
    pred_idx = 0
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            for z in range(grid.shape[2]):
                grid[x, y, z].state = predictions[pred_idx]
                pred_idx += 1
    print("Predicted next step:", predictions[:10])



class RLAgent:
    def __init__(self, action_space, state_space, learning_rate=0.1, gamma=0.9):
        self.action_space = action_space
        self.state_space = state_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.q_table = np.zeros((len(state_space), len(action_space)))

    def choose_action(self, state):
        if random.uniform(0, 1) < 0.1:  # Exploration
            return random.choice(self.action_space)
        else:  # Exploitation
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state, action] = (1 - self.learning_rate) * self.q_table[state, action] + \
                                      self.learning_rate * (reward + self.gamma * self.q_table[next_state, best_next_action])

def train_rl_agent(grid, agent):
    # Interacts with the grid and learns based on state transitions
    for episode in range(10):  # 10 times
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                for z in range(grid.shape[2]):
                    cell = grid[x, y, z]    
                    state = cell.state
                    action = agent.choose_action(state)
                    
                    next_state = (state + action) % 3  # Simplified state change
                    reward = -1 if next_state == 0 else 1  # Reward based on the action
                    
                    agent.update_q_table(state, action, reward, next_state)
                    cell.state = next_state  # Update cell state after the action
    print("Training completed using RL agent")

grid, blood_vessel_grid = create_simulation_grid()
distance_grid = calc_distance_vertical_vessel(grid.shape, blood_vessel_grid)

# Train models
knn_model = train_knn_predictive_model(grid)


# Main loop
for step in range(10):  # Run simulation for 10 steps
    print(f"Step {step + 1}")
    simulation_step(grid, blood_vessel_grid, cell_types=None, distance_grid=distance_grid, step=step)
    # Prediction using KNN
    predict_next_step_knn(grid, knn_model)


