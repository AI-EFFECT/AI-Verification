
import numpy as np
import pandas as pd
import yaml
import shutil
from pathlib import Path
from pyomo.environ import *

# --- Path Resolution ---
USECASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = USECASE_DIR.parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"
CONFIG_PATH = ROOT_DIR / "usecase/optimization/lp/config.yaml"

def save_pyomo_model_file():
    """
    Copies the current script logic or a dedicated pyomo file to the models folder.
    This allows the OptimizationLoader to import it dynamically.
    """
    MODELS_DIR.mkdir(exist_ok=True)
    dest_path = MODELS_DIR / "lp_physics.py"
    
    # We write the create_model and mapping functions to a file
    # This is what the user (or this script) defines as the problem
    content = """from pyomo.environ import *
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
    
    for i in range(4):
        expr = sum(A_out[i, j] * model.x_out[idx] for j, idx in enumerate(output_idxs))
        model.cons.add(expr <= b_raw[i])

    # ZONE 2: Global Coupling Constraints
    # This is where users define how inputs and outputs relate (e.g., Mass Balance)
    balance_expr = sum(model.x_in[i] for i in input_idxs) - \
                   sum(model.x_out[j] for j in output_idxs)
    model.cons.add(balance_expr <= 0)

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
"""
    with open(dest_path, "w") as f:
        f.write(content)
    return "models/lp_physics.py"

from scipy.optimize import linprog

def generate_data(n_samples=500):
    print("[*] Generating training data using Scipy Solver...")
    
    # 1. SETUP IDENTICAL PHYSICS (Matches Pyomo "baked" logic)
    np.random.seed(42)
    A_out = np.random.uniform(-1, 1, size=(4, 4))
    b_raw = np.random.uniform(5, 10, size=4)
    
    input_idxs = [0, 1]
    output_idxs = [2, 3, 4, 5]
    
    # Objective: Minimize sum(x_out) 
    # (Matches 'sum(model.x_out[j] for j in output_idxs)' in your Pyomo)
    c_output = np.ones(4) 
    
    data = []
    feasible_count = 0
    
    # 2. DATA GENERATION LOOP
    while feasible_count < n_samples:
        # Sample inputs within the specified range [1, 5]
        x_in = np.random.uniform(1, 5, size=len(input_idxs))
        
        # Balance Constraint: sum(x_in) <= sum(x_out)
        # In linprog (Ax <= b) form: -sum(x_out) <= -sum(x_in)
        A_balance = -np.ones((1, 4))
        b_balance = -np.sum(x_in)
        
        # Combine all constraints for the solver
        # A_out * x_out <= b_raw
        # A_balance * x_out <= b_balance
        A_combined = np.vstack([A_out, A_balance])
        b_combined = np.concatenate([b_raw, [b_balance]])
        
        # Solve
        res = linprog(c_output, A_ub=A_combined, b_ub=b_combined, 
                      bounds=(0, 10), method='highs')
        
        if res.success:
            # Row = [in_0, in_1, out_2, out_3, out_4, out_5]
            sample = np.concatenate([x_in, res.x])
            data.append(sample)
            feasible_count += 1
            if feasible_count % 100 == 0:
                print(f"    Generated {feasible_count}/{n_samples} samples...")

    # 3. SAVE OUTPUTS
    df = pd.DataFrame(data, columns=[f'in_{i}' for i in input_idxs] + [f'out_{j}' for j in output_idxs])
    df.to_csv(USECASE_DIR / "lp_data.csv", index=False)
    
    # Update meta to reflect the REAL physics used here
    # This helps the TRAIN script populate the config correctly
    meta = {
        "input_bounds": [{"min": 1, "max": 5}, {"min": 1, "max": 5}],
        "mapping": {
            "input_indices": input_idxs,
            "output_indices": output_idxs
        },
        "physics": {
            "A_out": A_out.tolist(),
            "b_raw": b_raw.tolist()
        }
    }
    with open(USECASE_DIR / "lp_metadata.yaml", "w") as f:
        yaml.dump(meta, f)

    print(f"[+] Successfully saved {n_samples} feasible samples to lp_data.csv")

def update_config_with_opt(opt_rel_path):
    """Initializes or updates the config.yaml with the path to the Pyomo model."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            data = yaml.safe_load(f)
    else:
        data = {"proxy_spec": {}}

    data["proxy_spec"]["opt_path"] = opt_rel_path
    
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

def main():
    print("[*] Starting Problem Generation...")
    # 1. Generate data for training
    generate_data()
    
    # 2. Save the Pyomo code to the models/ folder
    opt_path = save_pyomo_model_file()
    print(f"[+] Pyomo model saved to: {MODELS_DIR / 'lp_physics.py'}")
    
    # 3. Update the global config
    update_config_with_opt(opt_path)
    print(f"[+] Global config updated with opt_path.")

if __name__ == "__main__":
    main()






# from pyomo.repn import generate_standard_repn

# def analyze_and_extract(model):
#     # 1. Identify Type
#     # polynomial_degree(): 1=Linear, 2=Quadratic, None=Non-linear/Transcendental
#     degrees = []
#     for c in model.component_data_objects(Constraint):
#         degrees.append(c.body.polynomial_degree())
    
#     if None in degrees:
#         prob_type = "NLP (Non-linear)"
#     elif any(d > 1 for d in degrees if d is not None):
#         prob_type = "QCP (Quadratic)"
#     else:
#         prob_type = "LP (Linear)"
        
#     # 2. Extract Coefficients (The "A" matrix equivalent)
#     # For a verifier, you need to know which variable maps to which column
#     all_vars = list(model.component_data_objects(Var))
#     var_map = {id(v): i for i, v in enumerate(all_vars)}
    
#     matrix_A = []
#     vector_b = []
    
#     for c in model.component_data_objects(Constraint):
#         # This reduces the Pyomo expression to a standard numeric representation
#         repn = generate_standard_repn(c.body)
        
#         row = np.zeros(len(all_vars))
#         if repn.is_linear():
#             for v, coef in zip(repn.linear_vars, repn.linear_coefs):
#                 row[var_map[id(v)]] = coef
#         else:
#             # For NLPs, you would extract the partial derivatives (Jacobian)
#             # or use the symbolic string for SMT-based verification
#             pass 
            
#         matrix_A.append(row)
#         vector_b.append(value(c.upper) if c.has_ub() else value(c.lower))

#     return prob_type, np.array(matrix_A), np.array(vector_b)

# p_type, A, b = analyze_and_extract(model)
# print(f"Routing to {p_type} verifier...")
# print(A)
# print(b)




# def generate_agnostic_lp(n_vars=6, n_cons=4, num_samples=5000):
#     """
#     n_vars: Total variables in the canonical physics (x_full)
#     """
#     # 1. Generate Static Physics (A, b)
#     A = np.random.uniform(-1.0, 1.0, size=(n_cons, n_vars))
#     b = np.random.uniform(5.0, 10.0, size=n_cons)
    
#     # Define the Mapping Logic early so we can mask the objective
#     input_idxs = [0, 1]
#     output_idxs = [2, 3, 4, 5]
    
#     # --- ADDING THE BALANCE CONSTRAINT ---
#     # We want: sum(x_out) >= sum(x_in)  =>  sum(x_in) - sum(x_out) <= 0
#     balance_row = np.zeros((1, n_vars))
#     balance_row[0, input_idxs] = 1.0    # Load side
#     balance_row[0, output_idxs] = -1.0  # Generation side
    
#     A = np.vstack([A, balance_row])    # Add row to A
#     b = np.append(b, 0.0)              # Add 0 to b
#     # -------------------------------------

#     c = np.zeros(n_vars)
#     # Assign random costs ONLY to the output variables
#     c[output_idxs] = np.random.uniform(0.1, 1.0, size=len(output_idxs))

#     data = []

#     for i in range(num_samples):
#         # Sample random values for the 'input' subset of x (e.g. Demand levels)
#         x_in = np.random.uniform(1, 5, size=len(input_idxs))
        
#         # Physics Step: A_in * x_in + A_out * x_out <= b
#         # Rearranged:  A_out * x_out <= b - (A_in * x_in)
#         A_input = A[:, input_idxs]
#         A_output = A[:, output_idxs]
#         b_dynamic = b - (A_input @ x_in)
        
#         # Objective for the solver: minimize c_output^T * x_out
#         c_output = c[output_idxs]
        
#         # Solving the local LP for this specific input scenario
#         res = linprog(c_output, A_ub=A_output, b_ub=b_dynamic, bounds=(0, 10), method='highs')
        
#         if res.success:
#             # Store [Inputs, Optimal Outputs]
#             data.append(np.concatenate([x_in, res.x]))
#         else:
#             print("Infeasible!")

#     # 3. Save Data and Metadata
#     df = pd.DataFrame(data, columns=[f'in_{i}' for i in input_idxs] + [f'out_{j}' for j in output_idxs])
#     df.to_csv(BASE_DIR / "lp_data.csv", index=False)

#     metadata = {
#         "physics": {
#             "A": A.tolist(), 
#             "b": b.tolist(), 
#             "c": c.tolist() # This now reflects the zeroed-out input costs
#         },
#         "mapping": {
#             "input_indices": input_idxs, 
#             "output_indices": output_idxs
#         },
#         "input_bounds": [{"min": 0, "max": 5} for _ in input_idxs]
#     }
    
#     with open(BASE_DIR / "lp_metadata.yaml", "w") as f:
#         yaml.dump(metadata, f, sort_keys=False)
    
#     print(f"Successfully generated {len(data)} feasible samples.")

# if __name__ == "__main__":
#     generate_agnostic_lp()



