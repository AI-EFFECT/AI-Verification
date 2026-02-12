from pyomo.environ import *
import numpy as np

def create_model():
    '''
    ZONE 1: Define your optimization model.
    Users should define their Variables and Constants here.
    '''
    model = ConcreteModel()
    
    # Define which indices correspond to your Neural Network
    input_idxs = [0, 1]      # Features the NN receives
    output_idxs = [2, 3, 4, 5] # Decisions the NN predicts
    
    # --- VARIABLE DEFINITION ---
    # Define the bounds of your input space (the search space for the verifier)
    model.x_in = Var(input_idxs, bounds=(0, 5), initialize=0)
    
    # Define the bounds of the NN outputs
    model.x_out = Var(output_idxs, domain=NonNegativeReals, bounds=(0, 10))
    
    # --- PHYSICAL CONSTRAINTS ---
    model.cons = ConstraintList()
    
    # Example: Physical bounds or resource limits
    # Users can write these as natural algebraic expressions!
    np.random.seed(42) 
    A_out = np.random.uniform(-1, 1, size=(4, 4))
    b_raw = np.random.uniform(5, 10, size=4)
    
    # 1. Standard Upper Bound
    for i in range(4):
        expr = sum(A_out[i, j] * model.x_out[idx] for j, idx in enumerate(output_idxs))
        model.cons.add(expr <= b_raw[i])
        
    # 2. Lower Bound Constraint: Ensure at least some minimum activity
    # Example: sum of first two outputs must be >= 2.0
    model.cons.add(model.x_out[2] + model.x_out[3] >= 2.0)

    # 3. Range Constraint: Keep the last output within a specific window
    # Example: 1.0 <= x_out[5] <= 4.0
    model.cons.add((1.0, model.x_out[5], 4.0))

    # ZONE 2: Global Coupling Constraints
    # This is where users define how inputs and outputs relate (e.g., Mass Balance)
    balance_expr = sum(model.x_in[i] for i in input_idxs) -                    sum(model.x_out[j] for j in output_idxs)
    #model.cons.add(balance_expr <= 0)
    model.cons.add(balance_expr == 0)

    # ZONE 3: The Objective
    # This defines what "Good" looks like (used to calculate Regret/Optimality Gap)
    model.obj = Objective(expr=sum(model.x_out[j] for j in output_idxs), sense=minimize)
    
    return model

def get_io_mapping(model):
    '''
    CRITICAL: Map your Pyomo variables back to the Neural Network vector.
    The order here MUST match the order of your NN's input and output layers.
    '''
    inputs = [model.x_in[i] for i in [0, 1]]
    outputs = [model.x_out[j] for j in [2, 3, 4, 5]]
    return inputs, outputs
