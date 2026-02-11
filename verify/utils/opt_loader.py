# interface/opt_loader.py
import importlib.util
import sys
from pathlib import Path
from pyomo.repn import generate_standard_repn
import numpy as np
from pyomo.environ import Constraint, value

class OptimizationLoader:
    def __init__(self, config, root_dir: Path = None):
        self.config = config
        # Determine the root directory to find the /models folder
        if root_dir is None:
            # We are in ROOT/verify/nn_loader.py, so we need to go up ONE level
            self.root_dir = Path(__file__).resolve().parent.parent.parent
        else:
            self.root_dir = Path(root_dir).resolve()
            
        print(f"[*] Root directory identified as: {self.root_dir}")
        self.opt_path = self.root_dir / config['proxy_spec']['opt_path']
        
        # 1. Dynamically load the user's Pyomo file
        self.model_module = self._load_module()
        self.pyomo_model = self.model_module.create_model()
        self.inputs, self.outputs = self.model_module.get_io_mapping(self.pyomo_model)

    def _load_module(self):
        spec = importlib.util.spec_from_file_location("user_opt", self.opt_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["user_opt"] = module
        spec.loader.exec_module(module)
        return module

    def enrich_config(self):
        """Extracts A, b, c, and input bounds, then injects them into config."""
        from pyomo.environ import value, Objective, Constraint
        from pyomo.repn import generate_standard_repn
        
        all_vars = self.inputs + self.outputs
        var_map = {id(v): i for i, v in enumerate(all_vars)}
        
        input_ids = {id(v) for v in self.inputs}
        output_ids = {id(v) for v in self.outputs}
        
        # Build indices by comparing IDs, not the objects themselves
        in_idx = [i for i, v in enumerate(all_vars) if id(v) in input_ids]
        out_idx = [i for i, v in enumerate(all_vars) if id(v) in output_ids]
        
        # --- 1. Extract Input Bounds ---
        # Verification expects a list of tuples: [(min, max), (min, max)]
        extracted_bounds = []
        for v in self.inputs:
            # Pyomo bounds can be None if not set, so we provide defaults
            low = value(v.lb) if v.lb is not None else 0.0
            high = value(v.ub) if v.ub is not None else 1.0
            extracted_bounds.append({"min": float(low), "max": float(high)})
        
        # --- 1. Extract Constraints (A and b) ---
        A = []
        b_static = []
        for c in self.pyomo_model.component_data_objects(Constraint):
            repn = generate_standard_repn(c.body)
            row = [0.0] * len(all_vars)
            for v, coef in zip(repn.linear_vars, repn.linear_coefs):
                if id(v) in var_map:
                    row[var_map[id(v)]] = coef
            
            if c.has_ub():
                A.append(row)
                b_static.append(float(value(c.upper) - repn.constant))
            elif c.has_lb():
                A.append([-1 * val for val in row])
                b_static.append(float(-value(c.lower) + repn.constant))

        # --- Extract Objective (c vector) ---
        c_vector = [0.0] * len(all_vars)
        
        # Find the objective we defined: model.obj
        obj = next(self.pyomo_model.component_data_objects(Objective, active=True))
        
        # generate_standard_repn extracts coefficients WITHOUT needing variable values
        repn_obj = generate_standard_repn(obj.expr)
        
        for v, coef in zip(repn_obj.linear_vars, repn_obj.linear_coefs):
            var_id = id(v)
            if var_id in var_map:
                c_vector[var_map[var_id]] = float(coef)
        
        # Adjust for maximization if necessary
        if obj.sense == -1: # -1 is maximize
            c_vector = [-1.0 * val for val in c_vector]

        # --- 4. Inject into Config ---
        if 'verification_spec' not in self.config:
            self.config['verification_spec'] = {}
        
        # Inject the bounds so LPSpecParser finds them
        self.config['verification_spec']['input_bounds'] = extracted_bounds
        
        self.config['verification_spec']['constraints'] = {
            'A': np.array(A).tolist(),
            'b_static': np.array(b_static).tolist()
        }
        self.config['verification_spec']['objective_c'] = c_vector
        
        self.config['verification_spec']['indices'] = {
            'input_indices': in_idx,    
            'output_indices': out_idx  
        }

        return self.config