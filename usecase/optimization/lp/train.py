import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import yaml
import os
from pathlib import Path

# --- Path Resolution ---
# Current Location: AI_Verification/usecase/optimization/lp/train.py
# Root Location:    AI_Verification/
USECASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = USECASE_DIR.parent.parent.parent 
MODELS_DIR = ROOT_DIR / "models"

def load_metadata(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

import yaml
from pathlib import Path

def export_to_yaml(model, meta, config_path, model_filename):
    """Exports agnostic physics metadata and points to the saved model file."""
    # Ensure config_path is a Path object for easy checking
    config_file = Path(config_path)
    
    # 1. Load existing data if file exists, else initialize
    if config_file.exists():
        with open(config_file, 'r') as f:
            # Use safe_load to get existing dictionary
            # or handle the case where the file is empty (yielding None)
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    # 2. Ensure nested dictionaries exist so we don't get KeyErrors
    if "model_meta" not in data:
        data["model_meta"] = {}
    if "proxy_spec" not in data:
        data["proxy_spec"] = {}

    # 3. Update the data with your new values
    nn_rel_path = f"models/{model_filename}"

    # Update model_meta block
    data["model_meta"].update({
        "name": "lp_proxy",
        "pclass": "optimization",
        "ptype": "lp",
        "architecture": "feedforward",
        "activation": "relu",
        "check": "constraint",
        "report": "no",
        "solver": "gurobi",
        "engine": "milp"
    })

    # Update proxy_spec block
    data["proxy_spec"]["nn_path"] = nn_rel_path

    # 4. Save back to the yaml file
    with open(config_file, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

def main():
    # 0. Setup Root-level Folders
    MODELS_DIR.mkdir(exist_ok=True)
    
    model_name = "lp_example.pt"
    model_save_path = MODELS_DIR / model_name
    
    # Metadata is local to the usecase folder
    meta_path = USECASE_DIR / "lp_metadata.yaml"
    if not meta_path.exists():
        print(f"Metadata not found! Run generation script first.")
        return
    meta = load_metadata(meta_path)
    
    n_inputs = len(meta["mapping"]['input_indices'])
    n_outputs = len(meta["mapping"]['output_indices'])

    # 1. Load Data
    data_path = USECASE_DIR / "lp_data.csv"
    df = pd.read_csv(data_path)
    
    X_cols = [f'in_{i}' for i in meta["mapping"]['input_indices']]
    Y_cols = [f'out_{j}' for j in meta["mapping"]['output_indices']]
    
    X = torch.tensor(df[X_cols].values, dtype=torch.float32)
    Y = torch.tensor(df[Y_cols].values, dtype=torch.float32)

    # 2. Define & Train
    model = nn.Sequential(
        nn.Linear(n_inputs, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, n_outputs)
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print(f"[*] Training NN Surrogate...")
    for epoch in range(2001): 
        optimizer.zero_grad()
        loss = criterion(model(X), Y)
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0: print(f"    Epoch {epoch} | Loss: {loss.item():.6f}")

    # 3. Save to ROOT/models/
    torch.save(model, model_save_path)
    print(f"\n[+] Model saved to root models folder: {model_save_path}")

    # 4. Export Config to ROOT/config.yaml
    config_path = ROOT_DIR / "usecase/optimization/lp/config.yaml"
    export_to_yaml(model, meta, config_path, model_name)
    print(f"[+] Config exported to root: {config_path}")

if __name__ == "__main__":
    main()







# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# import yaml
# from pathlib import Path

# BASE_DIR = Path(__file__).resolve().parent

# def load_metadata(path):
#     with open(path, 'r') as f:
#         return yaml.safe_load(f)

# def export_to_yaml(model, meta, path):
#     """Exports model weights and agnostic physics metadata."""
#     mapping = meta["mapping"]
#     physics = meta["physics"]
    
#     data = {
#         "model_meta": {
#             "name": "lp_proxy",
#             "pclass": "optimization",
#             "ptype": "lp",
#             "architecture": "feedforward",
#             "activation": "relu",
#             "check": "constraint",
#             "report": "no",
#             "solver": "gurobi",
#             "engine": "milp"
#         },
#         "verification_spec": {
#             "input_bounds": meta["input_bounds"],
#             "indices": mapping,
#             "constraints": {
#                 "A": physics["A"],
#                 "b_static": physics["b"] 
#             },
#             "objective_c": physics["c"]
#         },
#         "layers": [],
        
#     }

#     for module in model:
#         if isinstance(module, nn.Linear):
#             data["layers"].append({
#                 "weights": module.weight.detach().numpy().tolist(),
#                 "biases": module.bias.detach().numpy().tolist()
#             })
            
#     with open(path, 'w') as f:
#         yaml.dump(data, f, sort_keys=False)

# def main():
#     # 0. Load Metadata
#     meta_path = BASE_DIR / "lp_metadata.yaml"
#     if not meta_path.exists():
#         print(f"Metadata not found! Run generation script first.")
#         return
#     meta = load_metadata(meta_path)
    
#     # Extract index mapping
#     mapping = meta["mapping"]
#     in_idx = mapping['input_indices']
#     out_idx = mapping['output_indices']
    
#     n_inputs = len(in_idx)
#     n_outputs = len(out_idx)

#     # 1. Load Data
#     data_path = BASE_DIR / "lp_data.csv"
#     if not data_path.exists():
#         print(f"Data not found at {data_path}!")
#         return
#     df = pd.read_csv(data_path)
    
#     # 2. Extract specific columns based on indices
#     X_cols = [f'in_{i}' for i in in_idx]
#     Y_cols = [f'out_{j}' for j in out_idx]
    
#     X = torch.tensor(df[X_cols].values, dtype=torch.float32)
#     Y = torch.tensor(df[Y_cols].values, dtype=torch.float32)

#     # 3. Define Architecture (Sized to the mapping)
#     model = nn.Sequential(
#         nn.Linear(n_inputs, 16),
#         nn.ReLU(),
#         nn.Linear(16, 16),
#         nn.ReLU(),
#         nn.Linear(16, n_outputs)
#     )

#     # 4. Train
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.MSELoss()

#     print(f"[*] Training NN Surrogate...")
#     print(f"    Input indices: {in_idx} (Size: {n_inputs})")
#     print(f"    Output indices: {out_idx} (Size: {n_outputs})")
    
#     for epoch in range(1001): 
#         model.train()
#         optimizer.zero_grad()
#         prediction = model(X)
#         loss = criterion(prediction, Y)
#         loss.backward()
#         optimizer.step()
        
#         if epoch % 200 == 0: 
#             print(f"    Epoch {epoch:4d} | Loss: {loss.item():.8f}")

#     # 5. Export
#     config_path = BASE_DIR / "config.yaml"
#     export_to_yaml(model, meta, config_path)
#     print(f"\n[+] Training complete. Config exported to: {config_path}")

# if __name__ == "__main__":
#     main()