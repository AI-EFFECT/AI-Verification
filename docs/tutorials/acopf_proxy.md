
# ⚛️ ACOPF Neural Network Verification Guide

This guide explains how to formally verify Neural Network proxies designed for the Alternating Current Optimal Power Flow (ACOPF) problem using this toolbox.

## 📖 Problem Formulation
The verifier is specialized for **Voltage-based Neural Networks**, as proposed in recent literature for improving worst-case guarantees. Unlike traditional proxies that predict generation directly, this model maps power demands to the complex voltage state, ensuring the system state is intrinsically defined.

* **Inputs:** Active Power Demand (Pd) and Reactive Power Demand (Qd) for all buses.
* **Outputs:** Real and Imaginary components of Voltage (Vr, Vi) for all buses. 

The verifier uses the predicted (Vr, Vi) to unroll the grid's physical laws, checking constraints across the entire continuous input uncertainty set.

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

---

## 📚 References & Methodology
The methodology implemented in this tool, specifically the unrolling of the Voltage NN for formal verification, is based on:

@article{giraud2025neural,
  title={Neural Networks for AC Optimal Power Flow: Improving Worst-Case Guarantees during Training},
  author={Giraud, Bastien and Nellikkath, Rahul and Vorwerk, Johanna and Alowaifeer, Maad and Chatzivasileiadis, Spyros},
  journal={arXiv preprint arXiv:2510.23196},
  year={2025}
}

***