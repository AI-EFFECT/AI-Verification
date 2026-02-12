# ⚛️ Neural Network LP Proxy: Logic & Architecture

This tutorial implements a **Neural Network Surrogate** for Linear Programming (LP) problems. It learns to predict optimal decision variables but can't explicitly enforce any constraints. Therefore, we need to rigorously verify whether the surrogate will violate any constraints across the whole input domain.

---

## 1. Problem Formulation

The optimization in its canonical form is written as follows. The goal is to minimize the operating cost of a system subject to linear constraints.

### The Math
$$
\begin{aligned}
\min_{x} \quad & c^T x \\
\text{s.t.} \quad & A_{ub} x \le b_{ub} \\
& A_{eq} x = b_{eq} \\
& l \le x \le u
\end{aligned}
$$



### Agnostic Variable Mapping
We partition the global state vector $x$ into two distinct subsets:
* **Inputs ($x_{in}$):** System state variables (e.g., nodal demand, renewable availability).
* **Outputs ($x_{out}$):** Decision variables predicted by the NN (e.g., generator setpoints, battery dispatch).

The Neural Network functions as the mapping: $f_{\theta}(x_{in}) \to x_{out}$.

---

## 2. Universal Constraint Coverage

To ensure the Neural Network respects physical reality, the system supports three fundamental types of linear constraints. These are automatically converted into the canonical $Ax \le b$ form during verification.

### A. Demand Satisfaction (Equalities)
In many cost-minimization problems, the mathematical "cheapest" solution is simply to do nothing ($x=0$). To prevent this, we enforce a **Balance** constraint:
> **Equation:** $\sum x_{out} = \sum x_{in}$

**Implementation:** This is unrolled into two inequalities: 
1. $(\sum x_{out} - \sum x_{in}) \le \epsilon$
2. $-(\sum x_{out} - \sum x_{in}) \le \epsilon$
This creates a "thin corridor" of feasibility, forcing the NN to balance the system perfectly.

### B. Safety & Operational Limits (Inequalities)
These represent the "hard walls" of the physical system that the NN must never cross.
* **Upper Bounds (UB):** e.g., $x_{out} \le \text{Max}$. Ensures equipment is not overloaded.
* **Lower Bounds (LB):** e.g., $x_{out} \ge \text{Min}$. Ensures generators do not operate in unstable regions.

### C. Operational Windows (Range Constraints)
Range constraints define a specific "safe zone" for a variable.
> **Logic:** $x_{l} \le x \le x_{u}$

**Implementation:** The system treats this as a simultaneous Upper Bound and Lower Bound check. If the NN predicts a value outside this window, the verifier identifies exactly which "side" of the window was violated.

---

## 3. The "Worst-Case" Verification Logic

The verification task is formulated as a **Maximum Violation Search**. Instead of checking random points, the MILP solver searches the entire continuous input space to find the specific $x_{in}$ that causes the Neural Network to break a physical law as severely as possible.

If the **Max Violation** across all rows of $A$ is less than our tolerance ($\epsilon = 10^{-6}$), the Neural Network is **Formally Proven** to be safe for all possible operating conditions within the defined input bounds.

---

## 4. Generate Data and Train Surrogate

As described previously, we need to follow the following three steps to configure the verification engine. 

1.  **Define Physics:** Write the underlying optimization problem in Pyomo and store it in the `models/` folder.

The `generate.py` script samples the input space $x_{in}$, solves the Pyomo LP for each sample to find the optimal $x_{out}$, and saves both the data for training and the `lp_physics.py` model for verification in the `models/` folder.


```bash
# 1. Generate the LP physics and data
python usecase/optimization/lp/generate.py

```

2.  **Save Weights:** Store your trained Neural Network as a `.pt` file in the `models/` folder.

For step 2, we need to train a neural network surrogate to learn the solution to our optimization problem. By running the `train_nn.py` script, we train our surrogate model, and store the weights as a .pt file in the `models/` folder.

```bash
# 2. Train the surrogate model
python usecase/optimization/lp/train.py
```

3.  **Configure Engine:** Fill in the remaining verification configuration parameters in `config.yaml`.

The final configuration parameters are filled in in the `config.yaml`, and we are good to go to run the verification!

```bash
# 2. Train the surrogate model
python main.py usecase/optimization/lp/config.yaml
```

---

## 5. What to Verify?

By leveraging the MILP engine, we can verify both constraint violations and the worst-case sub-optimality. With the crown engine, we can only verify the constraint violations. If the engine returns Safe, it means for all possible inputs, your NN stays within the physical limits. If Unsafe, the engine provides a counter-example: a specific input that causes your NN to give an illegal recommendation.





