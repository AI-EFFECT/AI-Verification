# 🛡️ Neural Network Verification Toolbox

Neural Networks are powerful surrogates but remain "black boxes" that cannot guarantee physical feasibility. In safety critical systems like power systems, a 1% error isn't just noise—it can be a system failure. This toolbox provides the formal proof that your NN will behave across the entire input domain, even in the worst-case scenarios.

---

## ⚡ Quickstart

Getting from a physical problem to a verified certificate is a streamlined three-step process. This toolbox automates the integration between the Pyomo optimization environment and the PyTorch learning framework.

1. **Define Physics:** Model your system's physical constraints and objectives using **Pyomo**. Once you have written the optimization in a `.py` file, store it in the `models/` folder. This model serves as the ground-truth for the verifier. Find an example for a Linear Programming Optimization probleme here: [models/lp_physics.py](./models/lp_physics.py).
2. **Trained Surrogate:** Upload your trained Neural Network model which serves as a surrogate model to your optimization problem. Currently, the toolbox only accepts Neural Networks trained in **PyTorch**. Save the trained weights as a `.pt` file in the `models/` folder.
3. **Formal Verification:** Configure your verification settings in `config.yaml` and run the entrypoint:

```bash
python main.py config.yaml
```

---

## 🚀 Overview
This service allows users to verify Neural Network predictions against physical constraints or optimal solutions. It supports two primary verification modes:

* **Sub-Optimality Analysis (`check: distance`):** Measures the "Optimality Gap"—how far a NN prediction is from the mathematically certain "True Optimal" solution.
* **Safety Analysis (`check: constraint`):** Identifies "Worst-Case Violations"—searching for the specific input that forces the NN to break physical boundaries (e.g., thermal limits, power balance).


## ⚙️ Core Engines

The toolbox features a dual-engine architecture, allowing users to balance the trade-off between mathematical precision and computational speed.

* **Exact Verification (MILP):** Transforms the Neural Network into a Mixed-Integer Linear Programming (MILP) formulation. It leverages high-performance solvers like **Gurobi** to provide a definitive, mathematical certificate. If a property is violated, it returns the exact **counter-example** (the specific input that broke the system).
    
* **Bound-Based Verification (CROWN):** Utilizes a state-of-the-art linear relaxation framework (**CROWN**) to propagate efficient symbolic bounds through the network. This provides a formal guarantee (lower/upper bounds) in a fraction of the time required for MILP. It is the preferred choice for large-scale architectures or rapid iterative testing where approximate certificates are sufficient.


## 📖 Documentation

For detailed guides and tutorials, refer to our documentation suite:

* **[Tutorial: Running an LP Example](./docs/tutorials/lp_proxy.md):** A comprehensive, step-by-step guide covering data generation, surrogate training, and executing your first verification.
* **[Configuration Guide](./docs/configuration.md):** A complete breakdown of all `config.yaml` parameters, from solver selection to engine-specific settings.

---

## 📊 Automated Reporting

Verification shouldn't be a "black box." Every time the engine runs, it generates a timestamped, interactive **Jupyter Notebook report** stored in the `/output` folder. This provides a transparent audit trail of the model's performance.

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




