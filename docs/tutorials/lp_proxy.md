# ⚛️ Neural Network LP Proxy: Logic & Architecture

This tutorial implements a **Neural Network Surrogate** for Linear Programming (LP) problems. It learns to predict optimal decision variables but can't explicitly enforce any constraints. Therefore, we need to rigorously verify whether the surrogate will violate any constraints across the whole input domain.

---

## 1. Problem Formulation

The optimization in its canonical form is written as follows. The goal is to minimize the operating cost of a system subject to linear constraints.

### The Math
$$
\begin{aligned}
\min_{x} \quad & c^T x \\
\text{s.t.} \quad & A_{ineq} x \le b_{ineq} \\
& A_{eq} x = b_{eq} \\
& x_{l} \le x \le x_{u}
\end{aligned}
$$



### Agnostic Variable Mapping
We partition the global state vector $x$ into two distinct subsets:
* **Inputs ($x_{in}$):** System state variables (e.g., nodal demand).
* **Outputs ($x_{out}$):** Decision variables predicted by the NN (e.g., generator setpoints).

The Neural Network functions as the mapping: $f_{\theta}(x_{in}) \to x_{out}$.

---

## 2. Constraint Handling

To ensure a Neural Network (NN) respects physical reality, the toolbox supports three fundamental types of linear constraints. These are automatically converted into the canonical $Ax \le b$ form during the verification process to define the **Feasible Region** of your model.

---

### A. Equalities (Conservation Laws)
Equalities are used when variables must match exactly—for example, ensuring that energy, mass, or cash flow is perfectly conserved. 
> **Logic:** $\sum \text{Outputs} = \sum \text{Inputs}$

**Pyomo Implementation:**

```python
balance_expr = sum(model.x_in[i] for i in input_idxs) - \
                   sum(model.x_out[j] for j in output_idxs)
model.cons.add(balance_expr == 0)
```

**Implementation:** This is unrolled into two inequalities: 
1. $(\sum x_{out} - \sum x_{in}) \le \epsilon$
2. $-(\sum x_{out} - \sum x_{in}) \le \epsilon$
This creates a "thin corridor" of feasibility, forcing the NN to balance the system perfectly.

### B. Safety Limits (Inequalities)
These represent the physical limits of the system that the NN must never violate.
* **Upper Bounds (UB):** e.g., "Do not exceed maximum battery capacity." ($x_{out} \le \text{Max}$.) 
* **Lower Bounds (LB):** e.g., "Maintain minimum grid frequency." ($x_{out} \ge \text{Min}$.) 

**Pyomo Implementation:**

```python
model.cons.add(model.x_out[2] + model.x_out[3] >= 2.0)
```

### C. Operational Limits (Box Constraints)
Box constraints are the most efficient way to define the "Safe Operating Range" for a specific variable. Instead of writing two separate rules for the floor and the ceiling, you define the entire window in one line.

> **Logic:** $x_{l} \le x \le x_{u}$

**Pyomo Implementation:**

```python
model.cons.add((1.0, model.x_out[5], 4.0))
```

**Implementation:** The system treats this as a simultaneous Upper Bound and Lower Bound check. If the NN predicts a value outside this window, the verifier identifies exactly which "side" of the window was violated.

---

## 3. The "Worst-Case" Verification Logic

The verification task is formulated as a **Maximum Violation Search**. Instead of checking random points, the MILP solver searches the entire continuous input space to find the specific $x_{in}$ that causes the Neural Network to break a physical law as severely as possible.

If the **Max Violation** across all rows of $A$ is less than our tolerance ($\epsilon = 10^{-6}$), the Neural Network is **Formally Proven** to be safe for all possible operating conditions within the defined input bounds.

---

## 4. Generate Data and Train Surrogate

The verification engine requires three key components to be synchronized in the `models/` directory: the physical constraints, the trained weights, and the configuration.

### Step 1: Define & Generate Physics
First, we define the underlying optimization problem in Pyomo. For this tutorial, the `generate.py` script performs two critical tasks:
1. It samples the input space $x_{in}$ and solves the Pyomo LP for each sample to find the optimal $x_{out}$ (generating your training labels).
2. It exports the `lp_physics.py` model, which the engine uses as the "ground truth" during verification.

You can run the script by executing the following command:

```bash
# 1. Generate the LP physics and data
python usecase/optimization/lp/generate.py

```

### Step 2: Train and Store the Neural Network
Next, we train a neural network surrogate to "learn" the behavior of the optimization solver. Running the `train.py` script will fit the model to the generated data and save the model state as a .pt file.

```bash
# 2. Train the surrogate model
python usecase/optimization/lp/train.py
```

### Step 3: Configure & Run the Engine

Finally, we link everything together in `config.yaml`. This file tells the engine where to find the weights, which physics rules to apply, and which safety properties to check.

Once configured, launch the verification:

```bash
# 3. Run the verification engine
python main.py usecase/optimization/lp/config.yaml
```

---

## 5. Interpreting Results

By leveraging the MILP engine, we can verify both constraint violations and the worst-case sub-optimality. With the crown engine, we can only verify the constraint violations. 

The engine provides a formal guarantee of your Surrogate's reliability:

* **Result: SAFE** ✅ 
    The engine has mathematically proven that for **every possible input** within your bounds, the Neural Network will never violate your physical constraints. You can deploy the surrogate with 100% confidence in its feasibility.
    
* **Result: UNSAFE** ❌ 
    The engine found a **Counter-Example**. It provides the specific input $x_{in}$ and the resulting illegal $x_{out}$. 
    * *Action:* Use this counter-example to perform **Adversarial Retraining** to "teach" the NN about this specific failure point.



