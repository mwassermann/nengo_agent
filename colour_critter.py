import grid
import nengo
import nengo.spa as spa
import numpy as np 


#we can change the map here using # for walls and RGBMY for various colours
mymap="""
#######
#  M  #
# # # #
# #B# #
#G Y R#
#######
"""

# defining the representational space
D = 32
vocab = spa.Vocabulary(D)
vocab.parse("WHITE+GREEN+RED+BLUE+MAGENTA+YELLOW")

#### Preliminaries - this sets up the agent and the environment ################ 
class Cell(grid.Cell):

    def color(self):
        if self.wall:
            return 'black'
        elif self.cellcolor == 1:
            return 'green'
        elif self.cellcolor == 2:
            return 'red'
        elif self.cellcolor == 3:
            return 'blue'
        elif self.cellcolor == 4:
            return 'magenta'
        elif self.cellcolor == 5:
            return 'yellow'
             
        return None

    def load(self, char):
        self.cellcolor = 0
        if char == '#':
            self.wall = True
            
        if char == 'G':
            self.cellcolor = 1
        elif char == 'R':
            self.cellcolor = 2
        elif char == 'B':
            self.cellcolor = 3
        elif char == 'M':
            self.cellcolor = 4
        elif char == 'Y':
            self.cellcolor = 5
            
            
world = grid.World(Cell, map=mymap, directions=int(4))
body = grid.ContinuousAgent()
world.add(body, x=1, y=2, dir=2)

#this defines the RGB values of the colours. We use this to translate the "letter" in 
#the map to an actual colour. Note that we could make some or all channels noisy if we
#wanted to
col_values = {
    0: [0.9, 0.9, 0.9], # White
    1: [0.2, 0.8, 0.2], # Green
    2: [0.8, 0.2, 0.2], # Red
    3: [0.2, 0.2, 0.8], # Blue
    4: [0.8, 0.2, 0.8], # Magenta
    5: [0.8, 0.8, 0.2], # Yellow
}

noise_val = 0.1 # how much noise there will be in the colour info

#You do not have to use spa.SPA; you can also do this entirely with nengo.Network()
model = spa.SPA()
with model:
    
    # create a node to connect to the world we have created (so we can see it)
    env = grid.GridNode(world, dt=0.005)

    ### Input and output nodes - how the agent sees and acts in the world ######

    #--------------------------------------------------------------------------#
    # This is the output node of the model and its corresponding function.     #
    # It has two values that define the speed and the rotation of the agent    #
    #--------------------------------------------------------------------------#
    def move(t, x):
        speed, rotation = x
        dt = 0.001
        max_speed = 20.0
        max_rotate = 10.0
        body.turn(rotation * dt * max_rotate)
        body.go_forward(speed * dt * max_speed)
        
    movement = nengo.Node(move, size_in=2)
    
    #--------------------------------------------------------------------------#
    # First input node and its function: 3 proximity sensors to detect walls   #
    # up to some maximum distance ahead                                        #
    #--------------------------------------------------------------------------#
    def detect(t):
        angles = (np.linspace(-0.5, 0.5, 3) + body.dir) % world.directions
        return [body.detect(d, max_distance=4)[0] for d in angles]
    proximity_sensors = nengo.Node(detect)

    #--------------------------------------------------------------------------#
    # Second input node and its function: the colour of the current cell of    #
    # agent                                                                    #
    #---------------------------------#
    def cell2rgb(t):
        
        c = col_values.get(body.cell.cellcolor)
        noise = np.random.normal(0, noise_val,3)
        c = np.clip(c + noise, 0, 1)
        
        return c
        
    current_color = nengo.Node(cell2rgb)
     
    #--------------------------------------------------------------------------#
    # Final input node and its function: the colour of the next non-whilte     #
    # cell (if any) ahead of the agent. We cannot see through walls.           #
    #--------------------------------------------------------------------------#
    def look_ahead(t):
        
        done = False
        
        cell = body.cell.neighbour[int(body.dir)]
        if cell.cellcolor > 0:
            done = True 
            
        while cell.neighbour[int(body.dir)].wall == False and not done:
            cell = cell.neighbour[int(body.dir)]
            
            if cell.cellcolor > 0:
                done = True
        
        c = col_values.get(cell.cellcolor)
        noise = np.random.normal(0, noise_val,3)
        c = np.clip(c + noise, 0, 1)
        
        return c
        
    ahead_color = nengo.Node(look_ahead)    
    
    ### Agent functionality - your code adds to this section ###################
    
    walldist = nengo.Ensemble(n_neurons=500, dimensions=3, radius=4)
    nengo.Connection(proximity_sensors, walldist)
    
    def movement_func(x):
        turn = x[2] - x[0]
        spd = x[1] - 0.5
        return spd, turn
        
    
    nengo.Connection(walldist, movement, function=movement_func)  

    # Function that maps noisy RGB input to a semantic pointer
    def rgb_to_pointer(x):
        x = np.clip(x, 0, 1)
        closest_color = min(col_values.keys(), key=lambda k: np.linalg.norm(np.array(col_values[k]) - x))
        
        color_names = ["WHITE", "GREEN", "RED", "BLUE", "MAGENTA", "YELLOW"]
        color_str = color_names[closest_color]
        
        # If the color is white, reduce its strength significantly
        if color_str == "WHITE":
            return vocab.parse(color_str).v * 0.3 
        else:
            return vocab.parse(color_str).v * 1.2 
    
    # Defining vision states to receive sensor input
    model.vision_close = spa.State(D, vocab=vocab)
    model.vision_far = spa.State(D, vocab=vocab)
    nengo.Connection(current_color, model.vision_close.input, function=rgb_to_pointer)
    nengo.Connection(ahead_color, model.vision_far.input, function=rgb_to_pointer)
    
     # Memory components with feedback
    model.memory = spa.State(D, vocab=vocab, feedback=0.3)
    model.buffer = spa.State(D, vocab=vocab, feedback=0.2)
    model.previous = spa.State(D, vocab=vocab, feedback=0.3)
    
    # Cleanup memory for learning associations
    model.cleanup = spa.AssociativeMemory(
        input_vocab=vocab,
        output_vocab=vocab,
        threshold=0.05)
    
    # Cleanup memory for predictions
    model.pred_cleanup = spa.AssociativeMemory(
        input_vocab=vocab,
        output_vocab=vocab,
        threshold=0.05)
    
    # Create a negative bias for white
    white_bias = nengo.Node(vocab.parse('WHITE').v * -0.3)
    nengo.Connection(white_bias, model.cleanup.input)
    nengo.Connection(white_bias, model.pred_cleanup.input)
    
    # Connect vision to memory
    nengo.Connection(model.vision_close.output, 
                    model.memory.input, 
                    transform=1.5,
                    synapse=0.03)
    
    # Update previous state from memory
    nengo.Connection(model.memory.output,
                    model.previous.input,
                    transform=1.5,
                    synapse=0.05)
    
    # Learning node to bind current color with next color (from vision_far)
    def learn_sequence(t, x):
        current = x[:D]
        next_color = x[D:2*D]
        
        white_vec = vocab.parse('WHITE').v
        current_white_sim = np.dot(current, white_vec)
        next_white_sim = np.dot(next_color, white_vec)
        
        # adaptive scaling for white/non-white
        if current_white_sim > 0.8 or next_white_sim > 0.8:
            return (current * next_color) * 2.0
        else:
            return (current * next_color) * 4.0
    
    learn_node = nengo.Node(learn_sequence, size_in=2*D)
    
    # Connect current color (memory) and next color (vision_far) to learning
    nengo.Connection(model.memory.output, learn_node[:D], transform=1.5)
    nengo.Connection(model.vision_far.output, learn_node[D:], transform=1.5)
    
    # Connect learning result to memory buffer
    nengo.Connection(learn_node, model.buffer.input, transform=2.0)
    
    # Connect buffer to cleanup
    nengo.Connection(model.buffer.output, model.cleanup.input, transform=2.0)
    
    # Prediction node - uses current and previous colors to predict next
    def make_prediction(t, x):
        current = x[:D]
        previous = x[D:2*D]
        
        white_vec = vocab.parse('WHITE').v
        current_white_sim = np.dot(current, white_vec)
        prev_white_sim = np.dot(previous, white_vec)
        
        # Use the pattern of previous->current to predict next
        if current_white_sim > 0.8 or prev_white_sim > 0.8:
            return (previous * current) * 2.0
        else:
            return (previous * current) * 4.0
    
    predict_node = nengo.Node(make_prediction, size_in=2*D)
    
    # Connect memory components to prediction node
    nengo.Connection(model.memory.output, predict_node[:D], transform=1.5)
    nengo.Connection(model.previous.output, predict_node[D:], transform=1.5)
    
    # Clean up and output prediction
    model.prediction = spa.State(D, vocab=vocab)
    nengo.Connection(predict_node, model.pred_cleanup.input, transform=2.0)
    nengo.Connection(model.pred_cleanup.output, model.prediction.input, transform=2.0)