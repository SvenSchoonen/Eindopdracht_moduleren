# The agent-based cellular automaton model to calc cancer growth. 
import json
import random 
import numpy as np

# cell types 

# load the parameters
def load_config(file_path="config.json"):
    with open(file_path, "r") as f:
        config = json.load(f)
    return config
   
config = load_config()

# apoptose, proliferatie, migratie, of rust. Mutatie en status en positie
class Cell:
    def __init__(self, parameters):
        self.cell_type = cell_type
        self.position = position  # Position in grid
        self.alive = True
        self.age = 0 # iteration
        self.mutation_count = 0  # Tracks the number of mutations the cell has undergone 

    def apoptose():
        return False
        
    def proliferatie(): # cel die deelt als er plek is
        pass 
        
    def rust():
        pass
        
    def migratie():
        zelf.position = newpostion # verplaatsen van een cell als er plek is
        
    def mutation_count():
        pass # mutatie wnr de cell muteerd en hoeveel mutaties
    
    def mutation():
        pass # cell muteerd
    def check_buren(): # check naasten in grid
        self.position = position  # [x, y, z] schuin is moet meer afstand hebben  [1,1,1] dus [1,1,2] [2,1,1] of [0,1,1], [1,1,0]
                                  # 
        
        
        
def initialize_grid(grid_size, cell_types, normal_ratio=0.5, tumor_ratio=0.05, stem_ratio=0.05):
    grid = np.empty(grid_size, dtype=object)
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            for z in range(grid_size[2]):
                rand_value = random.random() # make random number between 0 and one to 
                if rand_value < normal_ratio:
                    cell_type = cell_types["normal"]
                    print("normal cell made on", x, y, z)
                    grid[x,y,z] = cell_types['normal']
                if rand_value < tumor_ratio:
					print("tumor cell made on", x, y, z)
                    cell_type = cell_types["stem"]
                    grid[x,y,z] = cell_types['normal']
    print(grid)
    return 
print(config["cell_types"])
initialize_grid(config["grid_size"], config["cell_types"])
# Models ...
