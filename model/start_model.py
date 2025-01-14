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
        print("postiosn", position)
        self.alive = True
        self.age = 0 # iteration
        self.mutation_count = 0  # Tracks the number of mutations the cell has undergone 
        self.state = 0  # 0 = v1 1 = v2, 2= v3, 3 = v4

    def apoptose():
        return False
        
    def proliferatie(): # cel die deelt als er plek is
        pass 
        
    def rust():
        pass
        
    def migratie():
        zelf.position = newpostion # Kans op verplaatsen van een cell als er plek is
        
    def mutation_count():
        pass # mutatie wnr de cell muteerd en hoeveel mutaties
    
    def mutation():
        pass # cell muteerd
    def check_buren(): # check naasten in grid
        self.position = position  # [x, y, z] schuin is moet meer afstand hebben  [1,1,1] dus [1,1,2] [2,1,1] of [0,1,1], [1,1,0]
                                  # 
        
        
        
def initialize_grid(grid_size, cell_types, normal_ratio=0.9, tumor_ratio=0.1, stem_ratio=0.05): # aanmaak levensduur cel 
    grid = np.empty(grid_size, dtype=object)

    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            for z in range(grid_size[2]):
                rand_value = random.random() # make random number between 0 and one to 
                if rand_value < normal_ratio:
                    cell_type = cell_types["normal"]
                    #print("normal cell made on", x, y, z)
                    grid[x,y,z] = cell_types['normal']
                if rand_value < tumor_ratio:
                    #print("tumor cell made on", x, y, z)
                    grid[x,y,z] = cell_types['tumor']
                if rand_value < stem_ratio:
                    #print("Tumor stem cell made on", x, y, z)
                    grid[x,y,z] = cell_types['stem']
                # make blood vessel with heat map
                
                
                
    #print(grid)
    return 
# print(config["cell_types"])
  # Angiogenesis
def make_bloodvessel_grid(blood_vessel_place, grid_size, vessel_thickness): # Maybe add hor and vert
    print("start")
    grid_size = [6,6,6]
    blood_vessel_place = [2,2]   # going up in Z 
    x = blood_vessel_place[0]
    y = blood_vessel_place[1]
    blood_vessel_height = 9
    blood_vessel_thickness = [1, 1] # x , y , z   
        
    #print(grid)     
    return np.mgrid[:grid_size[0], :grid_size[1], :grid_size[2]].T.reshape((-1,vessel_thickness))	
print(make_bloodvessel_grid(config["vessel_place"], config["vessel_size"],  config["vessel_thickness"]))
initialize_grid(config["grid_size"], config["cell_types"])
# Models ...


