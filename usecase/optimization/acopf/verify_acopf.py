
import os
import sys
import numpy as np
import torch
import torch.nn.utils.prune as prune
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import yaml   
import time

# 1. Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
crown_path = os.path.join(project_root, "verify", "engines", "crown", "auto_LiRPA")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if crown_path not in sys.path:
    sys.path.insert(0, crown_path)

from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
from verify.models.acopf_augmented_model import OutputWrapper, NeuralNetwork
from verify.models import acopf_augmented_model
from verify.utils.loader import NNLoader

# Create an alias so the unpickler finds 'neural_network'
sys.modules['neural_network'] = acopf_augmented_model
sys.modules['neural_network.lightning_nn_crown'] = acopf_augmented_model

from .acopf_parameters import create_example_parameters
from types import SimpleNamespace

import os
import warnings
import logging

# 1. Suppress Python Warnings (like the auto_LiRPA UserWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*has batch dimension.*")

# 2. Suppress library loggers (INFO and WARNING from draw_layers, from_ppc, etc.)
# This target-mutes the specific loggers or sets a high threshold globaly
logging.getLogger().setLevel(logging.ERROR) 

# 3. Suppress the specific Numba warning via Environment Variable
os.environ['NUMBA_DISABLE_INTEL_SVML'] = '1' # General numba stability


def run_acopf_verification(loader, method_complex="alpha-CROWN", method_simple="backward"):
    
    # --- 1. Gatekeeper: Implementation & Compatibility Checks ---
    check_type = loader.meta.get('check', 'constraint')
    engine_type = loader.meta.get('engine', 'crown')

    if check_type == "distance":
        raise NotImplementedError("Distance (optimality gap) checks are not implemented for ACOPF.")
    
    if engine_type == "milp":
        raise NotImplementedError("ACOPF is not MILP compatible yet. Please use engine='crown'.")
    
    # 1. Parameter Extraction
    model_name = loader.meta.get('name', 'Unknown Model')
    n_buses = _extract_bus_count(model_name)
    sim_params = create_example_parameters(n_buses)
    
    # 2. Constraint Initialization
    sys = sim_params['true_system']
    map_g = torch.tensor(sys['Map_g'], dtype=torch.float32)
    n_gens = sim_params['general']['n_gbus']
    
    # 1. Pre-scale everything to avoid order-of-operation drifts
    sg_max_scaled = torch.tensor(sys['Sg_max']).float() / 100
    pg_min_scaled = torch.tensor(sys['pg_min']).float() / 100
    qg_min_scaled = torch.tensor(sys['qg_min']).float() / 100

    constraints = {
        'sd_min': torch.tensor(sys['Sd_min']).float() / 100,
        'sd_delta': torch.tensor(sys['Sd_delta']).float() / 100,
        'v_max': torch.tensor(sys['Volt_max'][0]).float(),
        'v_min': torch.tensor(sys['Volt_min'][0]).float(),
        'i_max': torch.cat([torch.tensor(sys['I_max_pu']).float()] * 2, dim=0),
        
        # Match original order: Scale @ Mapping
        'pg_max': ((sg_max_scaled.T @ map_g)[:, :n_buses]).squeeze(),
        'qg_max': ((sg_max_scaled.T @ map_g)[:, n_buses:]).squeeze(),
        
        'pg_min': (pg_min_scaled @ map_g[n_gens:, n_buses:]).squeeze(),
        'qg_min': (qg_min_scaled @ map_g[n_gens:, :n_buses]).squeeze()
    }

    
    gen_bus_mask = (constraints['pg_max'] > 0)
    
    # --- START TIMING ---
    start_time = time.perf_counter()

    # 3. Model & Input Setup
    nn_model = loader.load_from_file()
    x_min = constraints['sd_min'].view(1, -1)
    x_max = (constraints['sd_min'] + constraints['sd_delta']).view(1, -1)
    x = (x_min + x_max) / 2
    
    ptb = PerturbationLpNorm(x_L=x_min, x_U=x_max)
    bounded_input = BoundedTensor(x, ptb)
    opt_args = {'enable_alpha_crown': True, 'enable_beta_crown': True}

    # 4. Phase A: Intermediate McCormick Bounds
    IDX = {
        'vr': 0, 'vi': 1, 'ir': 2, 'ii': 3, 'imag': 4, 
        'vmag_up': 5, 'vmag_down': 6, 'pg_up': 7, 'pg_down': 8,
        'qg_up': 9, 'qg_down': 10, 'inj_r': 11, 'inj_i': 12
    }
    
    def get_bounds(idx, method="alpha-CROWN"):
        model = BoundedModule(OutputWrapper(nn_model, idx), torch.empty_like(x), opt_args)
        lb, ub = model.compute_bounds(x=(bounded_input,), method=method)
        return torch.min(lb, ub), torch.max(lb, ub)

    lb_vr, ub_vr = get_bounds(IDX['vr'])
    lb_vi, ub_vi = get_bounds(IDX['vi'])
    lb_ir, ub_ir = get_bounds(IDX['ir'])
    lb_ii, ub_ii = get_bounds(IDX['ii'])


    for predictor in [nn_model.pinj_upper_nn, nn_model.qinj_upper_nn, nn_model.pinj_lower_nn, nn_model.qinj_lower_nn]:
        predictor.update_mccormick_bounds(lb_vr, ub_vr, lb_vi, ub_vi, lb_ir, ub_ir, lb_ii, ub_ii)

    # 5. Phase B: Violation Metric Collection
    results = {}

    def add_violation(label, idx, limit, is_upper=True, mask=None, method="backward"):
        lb, ub = get_bounds(idx, method=method)
        lb, ub = lb.squeeze(), ub.squeeze()
        limit = limit.squeeze()
        viol = torch.relu(ub - limit) if is_upper else torch.relu(limit - lb)
        target = viol[mask] if mask is not None else viol
        results[f'{label} Max Violation'] = target.max().item()
        results[f'{label} Avg Violation'] = target.mean().item()
        return viol

    # Real Power (Pg)
    add_violation('Pg Up', IDX['pg_up'], constraints['pg_max'], mask=gen_bus_mask, method="alpha-CROWN")
    add_violation('Pg Down', IDX['pg_down'], constraints['pg_min'], is_upper=False, mask=gen_bus_mask, method="alpha-CROWN")

    # Reactive Power (Qg) - Added back
    add_violation('Qg Up', IDX['qg_up'], constraints['qg_max'], mask=gen_bus_mask, method="alpha-CROWN")
    add_violation('Qg Down', IDX['qg_down'], constraints['qg_min'], is_upper=False, mask=gen_bus_mask, method="alpha-CROWN")

    # Voltage (Vm)
    add_violation('Vm Up', IDX['vmag_up'], constraints['v_max'], method=method_simple)
    add_violation('Vm Down', IDX['vmag_down'], constraints['v_min'], is_upper=False, method=method_simple)

    # Current (Ibr) - Added back
    add_violation('Ibr tot', IDX['imag'], constraints['i_max'], method=method_simple)

    # Combine totals
    results['Pg tot Max Violation'] = max(results['Pg Up Max Violation'], results['Pg Down Max Violation'])
    results['Qg tot Max Violation'] = max(results['Qg Up Max Violation'], results['Qg Down Max Violation'])
    results['Vm tot Max Violation'] = max(results['Vm Up Max Violation'], results['Vm Down Max Violation'])
    
    # --- Current Balance (Ibal) Violation ---
    # We use alpha-CROWN here as requested for higher precision on residuals
    lb_r, ub_r = get_bounds(IDX['inj_r'], method=method_complex)
    lb_i, ub_i = get_bounds(IDX['inj_i'], method=method_complex)
    lb_r, ub_r = lb_r.squeeze(), ub_r.squeeze()
    lb_i, ub_i = lb_i.squeeze(), ub_i.squeeze()

    # Calculate worst-case magnitude: sqrt(max(lb^2, ub^2)_r + max(lb^2, ub^2)_i)
    # This represents the furthest the injection is from the zero-balance point.
    worst_case_inj_real = torch.max(lb_r**2, ub_r**2)
    worst_case_inj_imag = torch.max(lb_i**2, ub_i**2)
    
    # Adding a small epsilon before sqrt is a professional safety measure for numerical stability
    inj_violation = torch.sqrt(worst_case_inj_real + worst_case_inj_imag + 1e-9)

    results['Ibal tot Max Violation'] = inj_violation.max().item()
    results['Ibal tot Avg Violation'] = inj_violation.mean().item()
    
    # stop timer
    end_time = time.perf_counter()
    results['runtime_sec'] = end_time - start_time
    
    # Check if all collected metrics are valid finite numbers
    all_values_valid = all(
        isinstance(v, (int, float)) and np.isfinite(v) 
        for v in results.values()
    )

    if all_values_valid:
        results['status'] = "Success"
    else:
        results['status'] = "Numerical Failure"
        # Optional: log which field failed
        for k, v in results.items():
            if not np.isfinite(v):
                print(f"[!] Warning: {k} contains non-finite value: {v}")

    if results['status'] == "Success":
        _print_acopf_verification_report(results, results['runtime_sec'])
    else:
        print(f"Solver Error: {results['status']}")

    return results

def _extract_bus_count(name):
    match = re.search(r'(\d+)', name)
    if not match:
        raise ValueError(f"Invalid model name format: {name}")
    return int(match.group(1))

def _print_acopf_verification_report(results, runtime):
    """
    Specialized reporter for ACOPF verification results using CROWN/alpha-beta-CROWN.
    """
    # 1. Determine Overall Safety State
    # We consider the system 'SAFE' if the worst-case violation of all physical limits is 0.
    # Note: Ibal (Current Balance) is a residual, so we check if it's near zero.
    max_phys_viol = max(
        results.get('Pg tot Max Violation', 0.0),
        results.get('Qg tot Max Violation', 0.0),
        results.get('Vm tot Max Violation', 0.0),
        results.get('Ibr tot Max Violation', 0.0)
    )
    
    # Safety threshold for numerical noise in relaxations
    is_safe = max_phys_viol <= 1e-6 

    # --- HEADER ---
    print(f"\n{'#'*80}")
    if is_safe:
        print(f"### ACOPF VERIFICATION SUCCESS: REGION IS FORMALLY SAFE")
    else:
        print(f"### ACOPF VERIFICATION FAILURE: POTENTIAL VIOLATION DETECTED")
    
    print(f"### Solve Time: {runtime:.4f}s")
    print(f"{'#'*80}")

    # --- SUMMARY TABLE ---
    print(f"\n{'PHYSICAL LIMIT GROUP':<25} | {'MAX WORST-CASE VIOLATION':<25} | {'STATUS'}")
    print(f"{'-'*80}")
    
    groups = [
        ('Real Power (Pg)', 'Pg tot Max Violation'),
        ('Reactive Power (Qg)', 'Qg tot Max Violation'),
        ('Voltage Mag (Vm)', 'Vm tot Max Violation'),
        ('Branch Current (Ibr)', 'Ibr tot Max Violation'),
    ]

    for label, key in groups:
        val = results.get(key, 0.0)
        status = "[SAFE]" if val <= 1e-6 else "[VIOLATED]"
        print(f"{label:<25} | {val:<25.6f} | {status}")

    # --- FEASIBILITY RESIDUAL (Current Balance) ---
    print(f"\n{'KCL/FEASIBILITY RESIDUAL':<25} | {'VALUE'}")
    print(f"{'-'*80}")
    print(f"{'Max I-Balance Residual':<25} | {results.get('Ibal tot Max Violation', 0.0):.6f}")
    print(f"{'Avg I-Balance Residual':<25} | {results.get('Ibal tot Avg Violation', 0.0):.6f}")

    # --- DETAILED ANALYSIS ---
    print(f"\n{'DETAILED BREAKDOWN':<80}")
    print(f"{'='*80}")
    print(f"{'Constraint':<20} | {'Max Violation':<15} | {'Avg Violation':<15}")
    print(f"{'-'*80}")
    
    details = [
        ('Pg Upper Bound', 'Pg Up'),
        ('Pg Lower Bound', 'Pg Down'),
        ('Qg Upper Bound', 'Qg Up'),
        ('Qg Lower Bound', 'Qg Down'),
        ('Vm Upper Bound', 'Vm Up'),
        ('Vm Lower Bound', 'Vm Down')
    ]

    for label, prefix in details:
        m_v = results.get(f'{prefix} Max Violation', 0.0)
        a_v = results.get(f'{prefix} Avg Violation', 0.0)
        print(f"{label:<20} | {m_v:<15.6f} | {a_v:<15.6f}")

    # --- CROWN FOOTER ---
    print(f"\n{'='*80}")
    print("[*] METHOD: alpha-CROWN + McCormick Relaxations + Alpha-max Beta-min")
    print("    better results could be achieved with alpha-beta-CROWN, but this is not implemented yet.")
    print("[*] Note: These violations represent the formal 'worst-case' across the input")
    print("    uncertainty set. If Max Violation is 0, the grid is safe for all input points.")
    print(f"{'='*80}\n")













