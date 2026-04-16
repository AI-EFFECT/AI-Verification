
# ⚛️ ACOPF Neural Network Verification Guide

This guide explains how to formally verify Neural Network proxies designed for the Alternating Current Optimal Power Flow (ACOPF) problem using this toolbox.


## 📖 Problem Formulation
This verifier is built specifically for **Voltage-based Neural Networks** (Giraud et al., 2025). Unlike generation-based proxies, this architecture ensures the system state is physically defined by mapping demands to complex voltages.

* **Inputs ($x$):** Active ($P_d$) and Reactive ($Q_d$) power demand for all buses.
* **Outputs ($y$):** Real ($V_r$) and Imaginary ($V_i$) voltage components for all buses.

The verifier "unrolls" these voltages to all operational states to formally certify generation limits and branch flows.

## ⚠️ Compatibility & Requirements
The current implementation has the following strict dependencies:

* **Architecture:** Only **Voltage NNs** are supported. Models that predict generation ($P_g, Q_g$) or voltage polar coordinates ($|V|, \theta$) directly are incompatible with the current physics-unrolling logic.
* **Engine:** Optimized exclusively for **$\alpha, \beta$-CROWN**.
* **Naming Convention:** Files **must** follow the `acopf_XX_bus` format (e.g., `acopf_57_bus.pt`). The verifier extracts the integer `XX` to automatically configure the grid topology and constraint dimensions.


---

## 📂 File Setup
To run a verification task, you must place your grid data and model in the models/ directory:

1. **Grid Case:** Place the PGLib case file (e.g., .m file) in the models/ folder.
2. **NN Model:** Place your trained PyTorch model (.pt or .pth) in the models/ folder.

---

## ⚙️ Configuration
Create a config.yaml for your use case. Ensure the ptype is set to acopf to trigger the alpha-beta-CROWN reporting engine.

**Example Configuration:**

```yaml
model_meta:
  name: acopf_57_bus
  pclass: optimization
  ptype: acopf
  architecture: feedforward
  activation: relu
  check: constraint
  report: 'yes'
  solver: gurobi
  engine: crown
proxy_spec:
  nn_path: models/acopf_example.pt
```
---

## 🚀 Running the Verification

1. **Activate Environment**
Ensure you are in the environment where auto_LiRPA and pytorch-lightning are installed:
conda activate verif_aie

2. **Execute**
Run the main entry point pointing to your ACOPF config:
python main.py usecase/optimization/acopf/config.yaml

---

## 📊 Understanding the Output
The tool generates a Jupyter Notebook Report in the output/ folder utilizing formal interval bounds.

* **Certified Safe:** If the "Max Violation" is 0.000000, the NN is mathematically guaranteed to be safe for all load scenarios within your bounds.
* **Potential Violation:** If the value is > 10^-6, the report highlights which physical limit (e.g., Pg Upper Bound or Vm Lower Bound) is at risk.
* **Feasibility Residual:** The I-Balance table shows the KCL mismatch, indicating how physically consistent the predicted voltages are with the current loads.

For an example output report, please check: [ACOPF Report](../../output/report_constraint_20260416_113344.ipynb)

---

## 📚 References & Methodology
The methodology implemented in this tool, specifically the unrolling of the Voltage NN for formal verification, is based on:

```bibtex
@article{giraud2025neural,
  title={Neural Networks for AC Optimal Power Flow: Improving Worst-Case Guarantees during Training},
  author={Giraud, Bastien and Nellikkath, Rahul and Vorwerk, Johanna and Alowaifeer, Maad and Chatzivasileiadis, Spyros},
  journal={arXiv preprint arXiv:2510.23196},
  year={2025}
}
```

***