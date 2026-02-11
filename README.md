# 🛡️ Neural Network Verification Toolbox

A dockerized service for formal verification of Neural Networks embedded in physical systems. This toolbox bridges the gap between machine learning models and mathematical optimization, providing safety and optimality guarantees.

---

## 🚀 Overview
This service allows users to verify Neural Network predictions against physical constraints or optimal solutions. It supports two primary verification modes:

* **Sub-Optimality Analysis (`check: distance`):** Measures the "Optimality Gap"—how far a NN prediction is from the mathematically certain "True Optimal" solution.
* **Safety Analysis (`check: constraint`):** Identifies "Worst-Case Violations"—searching for the specific input that forces the NN to break physical boundaries (e.g., thermal limits, power balance).



---

## 🛠️ Project Structure
```text
.
├── main.py                 # Service Entrypoint & Agnostic Router
├── config.yaml             # Single Source of Truth (Problem Definition)
├── usecase/                # Problem Class Logic (optimization, control, forecast)
├── verify/                 # Core Verification Engines (MILP, CROWN)
├── output/
│   ├── reports.ipynb       # Generated timestamped .ipynb files
│   └── utils/              # Report Generation Logic (nbformat)
└── Dockerfile              # Containerization for reproducible environments (to be implemented)


## 🚀 Getting Started

To get started, please follow the instructions below.

```text

```

## ⚙️ Configuration Reference

The behavior of the toolbox is governed by the `model_meta` section in your `config.yaml`. This section acts as the "brain" for the agnostic router and the reporting engine.

### 📝 Metadata Breakdown

| Key | Example | Description |
| :--- | :--- | :--- |
| `name` | `"lp_proxy"` | A unique identifier for the model. This string is used in the report headers and file naming. |
| `pclass` | `"optimization"` | **Routing Class:** Determines the high-level logic folder. Currently supports `optimization`, with hooks for `control` or `forecast` in development. |
| `ptype` | `"lp"` | **Problem Type:** Specifies the mathematical structure (e.g., Linear Programming, Mixed-Integer, or Quadratic). This directs the system to the correct solver interface. |
| `check` | `"constraint"` | **Verification Mode:** <br>• `constraint`: Searches for safety violations (worst-case $Ax > b$). <br>• `distance`: Measures the optimality gap against the ground-truth. |
| `report` | `"yes"` | **Reporting Toggle:** Accepts `yes`/`no` or `true`/`false`. If enabled, a Jupyter Notebook is automatically generated in the `output/` directory. |
| `architecture` | `"feedforward"`| **Model Type:** Informs the `NNLoader` how to parse the weights (e.g., `feedforward` for MLPs or `cnn` for convolutional nets). |
| `engine` | `"milp"`| **Verification Engine:** Selects the solver. `milp` is complete (provides exact worst-case results) but slow. `crown` is incomplete (provides a fast formal upper bound) and may include a relaxation gap. |
| `solver` | `"gurobi"`| **Verification Solver:** Selects the solver for the `milp` verification. |
| `activation` | `"relu"` | **Activation Function:** Specifies the non-linearity. This is vital for MILP-based verification, which requires specific encodings for `relu` vs `sigmoid`. |

---

