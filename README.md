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

To get started, follow one of the tutorials under the /docs folder.


---

