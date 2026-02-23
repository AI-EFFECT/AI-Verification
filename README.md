# 🛡️ Neural Network Verification Toolbox

Neural Networks are powerful surrogates but remain "black boxes" that cannot guarantee physical feasibility. In safety critical systems like power systems, a 1% error isn't just noise—it can be a system failure. This toolbox provides the formal proof that your NN will behave across the entire input domain, even in the worst-case scenarios.

---
## 🧠 Verification in a Nutshell

Traditional testing evaluates a Neural Network at specific points, but it cannot prove what happens between those points. Neural Network Verification treats the model as a mathematical function and searches the entire continuous input space to find the absolute worst-case scenario. Instead of asking "Does it work on my test set?", we ask: "Is there any possible input where this model violates physics?" By using MILP and CROWN engines, this toolbox provides a formal certificate that no such failure exists, or identifies the exact counter-example if it does.


## ⚡ Quickstart

Bridge the gap between **PyTorch** and **Pyomo** in minutes. This toolbox automates the generation of a formal safety certificate for your neural network surrogates.

### 1. Define Your Physics
Model your system's constraints and objectives using standard **Pyomo** syntax. Save this as a `.py` file in the `models/` folder. This acts as the "Ground Truth" for the verifier. *See a Linear Programming example:* [`lp_physics.py`](./models/lp_physics.py)

### 2. Connect Your Surrogate
Provide your trained Neural Network weights (currently supporting **PyTorch** `.pt` files). Place your model in the `models/` folder alongside your physics definition.

### 3. Formally Verify
Define your verification parameters in `config.yaml`, then trigger the verification toolbox:

<details>
<summary>🔍 Click to view sample Config structure</summary>

```yaml
model_meta:
  name:         lp_proxy
  pclass:       optimization
  ptype:        lp
  architecture: feedforward
  activation:   relu
  check:        constraint    # 'constraint' (safety) or 'distance' (optimality)
  report:       yes           # Generates an interactive Jupyter audit of the worst-case failure in /output
  solver:       gurobi        # MILP solver
  engine:       milp          # MILP or CROWN engine
proxy_spec:
  nn_path:      models/lp_example.pt
  opt_path:     models/lp_physics.py
```
</details>

```bash
python main.py config.yaml
```

## 📊 Verification Modes

If `report` is set to `yes`, this toolbox automates the generation of a formal audit report by identifying the **global maximum violation**—the exact point where your surrogate model will fail.

| Feature | **Safety Analysis** (`constraint`) | **Optimality Analysis** (`distance`) |
| :--- | :--- | :--- |
| **Focus** | Physical Feasibility | Sub-Optimality |
| **The Question** | Does the NN break physical laws? | How much "money" is left on the table? |
| **Metric** | Max Violation (e.g., $10^{-6}$) | Sub-Optimality Gap ($Cost_{NN} - Cost_{Opt}$) |
| **Example** | [View Constraint Report](./output/report_constraint_20260212_144036.ipynb) | [View Distance Report](./output/report_distance_20260212_144050.ipynb) |

> [!NOTE]
> All results are derived from a **Snapshot Inference**: the verifier identifies the single worst-case point in the continuous input space and reports the system state at that specific failure.


## ⚙️ Core Engines

The toolbox features a dual-engine architecture, allowing users to balance the trade-off between mathematical precision and computational speed.

* **Exact Verification (MILP):** Transforms the Neural Network into a Mixed-Integer Linear Programming (MILP) formulation. It leverages high-performance solvers like **Gurobi** to provide a definitive, mathematical certificate. If a property is violated, it returns the exact **counter-example** (the specific input that broke the system). Supports both the `constraint` and `distance` check.
    
* **Bound-Based Verification (CROWN):** Utilizes a state-of-the-art linear relaxation framework (CROWN) to propagate efficient symbolic bounds through the network. This provides a formal guarantee (lower/upper bounds) in a fraction of the time required for MILP. It is the preferred choice for large-scale architectures or rapid iterative testing where approximate certificates are sufficient. Only supports the `constraint` check.


## 📖 Documentation

For detailed guides and tutorials, refer to our documentation suite:

* **[Tutorial: Running an LP Example](./docs/tutorials/lp_proxy.md):** A comprehensive, step-by-step guide covering data generation, surrogate training, and executing your first verification.
* **[Configuration Guide](./docs/configuration.md):** A complete breakdown of all `config.yaml` parameters, from solver selection to engine-specific settings.
* **[Verification Theory](./docs/theory.md):** An explanation of how MILP and CROWN provide formal guarantees.

---

## 🛠️ Project Structure
```text
.
├── main.py                # Service Entrypoint & Agnostic Router
├── config.yaml            # Single Source of Truth (Problem Definition)
├── README.md              # Project Overview & Documentation
├── docs/                  # Detailed Tutorials & Technical Logic
├── models/                # User Storage: .pt weights & Pyomo physics files
├── usecase/               # Problem Class Logic (optimization, control, forecast)
├── verify/                # Core Verification Engines (MILP, CROWN)
├── output/                # Generated timestamped .ipynb reports
└── Dockerfile             # Containerization for reproducible environments
```




---

## 📚 References & Acknowledgments

This toolbox utilizes the [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) library for robust neural network bounding and formal verification.