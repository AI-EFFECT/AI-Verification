# ⚛️ Neural Network LP Proxy: Logic & Architecture

This tutorial implements a Neural Network Surrogate for Linear Programming (LP) problems. While NNs are excellent at approximating solvers, they cannot natively enforce hard physical constraints. This toolbox provides the formal verification necessary to prove safety across the entire input domain.

---
# 📘 Part 1: Theoretical Framework
*Understanding the math behind the verification.*

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

**Under the hood:** This is unrolled into two inequalities: 
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
---

# 🚀 Part 2: Hands-on Tutorial (LP Proxy)
*Follow these steps to generate data, train your model, and verify it.*

## 4. Tutorial Start - Generate Data, Train Proxy and Verify

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

> [!NOTE]: This script generates the following physics definition file for you, and places it in the correct folder.

<details>
<summary>🔍 View Pyomo Model Definition (Python Example)</summary>

```python
from pyomo.environ import *
import numpy as np

def create_model():
    '''
    ZONE 1: Define your optimization model.
    Users should define their Variables and Constants here.
    '''
    model = ConcreteModel()
    
    # Define which indices correspond to your Neural Network
    input_idxs = [0, 1]      # Features the NN receives
    output_idxs = [2, 3, 4, 5] # Decisions the NN predicts
    
    # --- VARIABLE DEFINITION ---
    # Define the bounds of your input space (the search space for the verifier)
    model.x_in = Var(input_idxs, bounds=(0, 5), initialize=0)
    
    # Define the bounds of the NN outputs
    model.x_out = Var(output_idxs, domain=NonNegativeReals, bounds=(0, 10))
    
    # --- PHYSICAL CONSTRAINTS ---
    model.cons = ConstraintList()
    
    # Example: Physical bounds or resource limits
    # Users can write these as natural algebraic expressions!
    np.random.seed(42) 
    A_out = np.random.uniform(-1, 1, size=(4, 4))
    b_raw = np.random.uniform(5, 10, size=4)
    
    # 1. Standard Upper Bound
    for i in range(4):
        expr = sum(A_out[i, j] * model.x_out[idx] for j, idx in enumerate(output_idxs))
        model.cons.add(expr <= b_raw[i])
        
    # 2. Lower Bound Constraint: Ensure at least some minimum activity
    # Example: sum of first two outputs must be >= 2.0
    model.cons.add(model.x_out[2] + model.x_out[3] >= 2.0)

    # 3. Range Constraint: Keep the last output within a specific window
    # Example: 1.0 <= x_out[5] <= 4.0
    model.cons.add((1.0, model.x_out[5], 4.0))

    # ZONE 2: Global Coupling Constraints
    # This is where users define how inputs and outputs relate (e.g., Mass Balance)
    balance_expr = sum(model.x_in[i] for i in input_idxs) - \
                   sum(model.x_out[j] for j in output_idxs)
    #model.cons.add(balance_expr <= 0)
    model.cons.add(balance_expr == 0)

    # ZONE 3: The Objective
    # This defines what "Good" looks like (used to calculate Regret/Optimality Gap)
    model.obj = Objective(expr=sum(model.x_out[j] for j in output_idxs), sense=minimize)
    
    return model

def get_io_mapping(model):
    '''
    CRITICAL: Map your Pyomo variables back to the Neural Network vector.
    The order here MUST match the order of your NN's input and output layers.
    '''
    inputs = [model.x_in[i] for i in [0, 1]]
    outputs = [model.x_out[j] for j in [2, 3, 4, 5]]
    return inputs, outputs
```

</details>

### Step 2: Train and Store the Neural Network
Next, we train a neural network surrogate to "learn" the behavior of the optimization solver. Running the `train.py` script will fit the model to the generated data and save the model state as a .pt file in the correct folder.

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



