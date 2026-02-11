import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yaml

def generate_agnostic_nlp(num_samples=1000):
    # Mapping
    input_idxs = [0, 1]      # e.g., Nodal Demands (P_load)
    output_idxs = [2, 3, 4]  # e.g., Gen outputs (P_gen) or Voltages
    
    # 1. Define the Nonlinear Physics (Standard Form)
    def objective(x_out, x_in):
        # Quadratic cost: sum(c2 * Pgen^2 + c1 * Pgen)
        return np.sum(0.1 * x_out**2 + 0.5 * x_out)

    def equality_constraints(x_out, x_in):
        # e.g., Power Balance: Sum(Gen) - Sum(Load) - Losses = 0
        # Simple nonlinear loss approximation: 0.05 * x_out^2
        losses = 0.05 * np.sum(x_out**2)
        return np.sum(x_out) - np.sum(x_in) - losses

    def inequality_constraints(x_out, x_in):
        # e.g., Standard NLP constraints like voltage limits or line flows
        # For now, a generic nonlinear limit: x_in * x_out <= limit
        return 20.0 - (x_in[0] * x_out[0] + x_out[1]**2)

    data = []
    for i in range(num_samples):
        # Sample Inputs (Demands)
        x_in_sampled = np.random.uniform(1, 5, size=len(input_idxs))
        
        # Initial guess for outputs
        x0 = np.ones(len(output_idxs)) * 2
        
        # Scipy Minimize Setup
        cons = [
            {'type': 'eq', 'fun': equality_constraints, 'args': (x_in_sampled,)},
            {'type': 'ineq', 'fun': inequality_constraints, 'args': (x_in_sampled,)}
        ]
        
        res = minimize(
            objective, x0, args=(x_in_sampled,),
            constraints=cons, bounds=[(0, 10)] * len(output_idxs)
        )
        
        if res.success:
            data.append(np.concatenate([x_in_sampled, res.x]))

    # Save as before...