# 🛡️ Neural Network Verification Toolbox

Neural Networks are powerful surrogates but remain "black boxes" that cannot guarantee physical feasibility. In safety critical systems like power systems, a 1% error isn't just noise—it can be a system failure. This toolbox provides the formal proof that your NN will behave across the entire input domain, even in the worst-case scenarios.

---

## ⚡ Quickstart

Bridge the gap between **PyTorch** and **Pyomo** in minutes. This toolbox automates the generation of a formal safety certificate for your neural network surrogates.

### 1. Define Your Physics
Model your system's constraints and objectives using standard **Pyomo** syntax. Save this as a `.py` file in the `models/` folder. This acts as the "Ground Truth" for the verifier.
* *See a Linear Programming example:* [`lp_physics.py`](./models/lp_physics.py)

### 2. Connect Your Surrogate
Provide your trained Neural Network weights (currently supporting **PyTorch** `.pt` files). Place your model in the `models/` folder alongside your physics definition.

### 3. Formally Verify
Adjust your search bounds and tolerance in `config.yaml`, then trigger the MILP-based verification engine:

```bash
python main.py config.yaml
```

## 📊 Verification Modes

This toolbox automates the generation of a formal audit report by identifying the **global maximum violation**—the exact point where your surrogate model will fail.

| Feature | **Safety Analysis** (`constraint`) | **Optimality Analysis** (`distance`) |
| :--- | :--- | :--- |
| **Focus** | Physical Feasibility | Sub-Optimality / Regret |
| **The Question** | Does the NN break physical laws? | How much "money" is left on the table? |
| **Metric** | Max Violation (e.g., $10^{-6}$) | Sub-Optimality Gap ($Cost_{NN} - Cost_{Opt}$) |
| **Example** | [View Constraint Report](./output/report_constraint_20260212_144036.ipynb) | [View Distance Report](./output/report_distance_20260212_144050.ipynb) |

> [!NOTE]
> All results are derived from a **Snapshot Inference**: the verifier identifies the single worst-case point in the continuous input space and reports the system state at that specific failure.


## ⚙️ Core Engines

The toolbox features a dual-engine architecture, allowing users to balance the trade-off between mathematical precision and computational speed.

* **Exact Verification (MILP):** Transforms the Neural Network into a Mixed-Integer Linear Programming (MILP) formulation. It leverages high-performance solvers like **Gurobi** to provide a definitive, mathematical certificate. If a property is violated, it returns the exact **counter-example** (the specific input that broke the system). Supports both the `constraint` and `distance` check.
    
* **Bound-Based Verification (CROWN):** Utilizes a state-of-the-art linear relaxation framework (**CROWN**) to propagate efficient symbolic bounds through the network. This provides a formal guarantee (lower/upper bounds) in a fraction of the time required for MILP. It is the preferred choice for large-scale architectures or rapid iterative testing where approximate certificates are sufficient. Only supports the `constraint` check.


## 📖 Documentation

For detailed guides and tutorials, refer to our documentation suite:

* **[Tutorial: Running an LP Example](./docs/tutorials/lp_proxy.md):** A comprehensive, step-by-step guide covering data generation, surrogate training, and executing your first verification.
* **[Configuration Guide](./docs/configuration.md):** A complete breakdown of all `config.yaml` parameters, from solver selection to engine-specific settings.

---

## 📊 Automated Reporting

Every time the engine runs, and you set `report: yes` in the `config.yaml`, it generates a timestamped, interactive **Jupyter Notebook report** stored in the `/output` folder. This provides a transparent audit trail of the model's performance.

**Each report includes:**
* **Violation Analytics:** Magnitude and frequency of worst-case constraint violations.
* **Optimality Gap:** Statistical distribution of the sub-optimality distance from the true mathematical optimum.
* **Counter-Example Visualizations:** High-resolution plots of the specific input scenarios that caused the Neural Network to fail, allowing for targeted model retraining.

**[Output Report Example](./output/report_constraint_20260212_111118.ipynb):** An output report example if you run a constraint verification.

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




