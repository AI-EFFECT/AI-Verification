from pyomo.environ import *
import numpy as np

def create_model():
    model = ConcreteModel()
    input_idxs = [0, 1]
    output_idxs = [2, 3, 4, 5]
    
    # NN Inputs 
    model.x_in = Var(input_idxs, bounds=(0,5), initialize=0)
    # NN Outputs 
    model.x_out = Var(output_idxs, domain=NonNegativeReals, bounds=(0, 10))
    
    # 1. Random linear constraints (identical to your linprog A_output)
    np.random.seed(42) 
    A_out = np.random.uniform(-1, 1, size=(4, 4))
    b_raw = np.random.uniform(5, 10, size=4)
    
    model.cons = ConstraintList()
    for i in range(4):
        expr = sum(A_out[i, j] * model.x_out[idx] for j, idx in enumerate(output_idxs))
        model.cons.add(expr <= b_raw[i])

    # 2. The Balance Constraint (Replacing the NL one)
    # sum(x_in) - sum(x_out) <= 0
    balance_expr = sum(model.x_in[i] for i in input_idxs) - sum(model.x_out[j] for j in output_idxs)
    model.cons.add(balance_expr <= 0)

    model.obj = Objective(expr=sum(model.x_out[j] for j in output_idxs), sense=minimize)
    return model

def get_io_mapping(model):
    # Map Pyomo objects to NN indices
    inputs = [model.x_in[i] for i in [0, 1]]
    outputs = [model.x_out[j] for j in [2, 3, 4, 5]]
    return inputs, outputs
