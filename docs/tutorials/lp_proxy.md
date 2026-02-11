# ⚛️ Neural Network LP Proxy: Logic & Architecture

This example implements a **Neural Network Surrogate** for Linear Programming (LP) problems. It learns to predict optimal decision variables but can't explicitly enforce any constraints. Therefore, we need to rigorously verify whether the surrogate will violate any constraints across the whole input domain.

---

## 1. Problem Formulation

The optimization in it's canonical form can be written as follows. The goal is to minimize the operating cost of a system subject to linear inequalities.

### The Math
$$
\begin{aligned}
\min_{x} \quad & c^T x \\
\text{s.t.} \quad & A x \le b \\
& x \ge 0
\end{aligned}
$$

### Agnostic Variable Mapping
We partition the global state vector $x$ into two distinct subsets:
* **Inputs ($x_{in}$):** Some system input variables (e.g., nodal demand, weather data).
* **Outputs ($x_{out}$):** Decision variables predicted by the NN (e.g., generator setpoints, battery dispatch).

The Neural Network functions as the mapping: $f_{\theta}(x_{in}) \to x_{out}$.

In many cost-minimization problems, the mathematical "cheapest" solution is simply to do nothing ($x=0$). To prevent this and simulate a real-world system, we enforce a **Demand Satisfaction** constraint:

> **Equation:** $\sum x_{out} \ge \sum x_{in}$

This forces the Neural Network to learn the "Load Following" logic—it must increase output variables whenever the input demand increases. This is injected into the global $A$ matrix and $b$ vector before training and verification.

---

## 2. Generate Data and Train Surrogate

As described previously, we need to follow the following three steps to configure the verification engine. 

1.  **Define Physics:** Write the underlying optimization problem in Pyomo and store it in the `models/` folder.

For step 1, we formulate our optimization problem, and store it as a pyomo model using the `generate_data.py` script. By running this script, we solve many instances of the underlying optimization problem to train our neural network, and store the pyomo model in the /models folder.

```bash
# 1. Generate the LP physics and data
python usecase/optimization/lp/generate.py

```

2.  **Save Weights:** Store your trained Neural Network as a `.pt` file in the `models/` folder.

For step 2, we need to train a neural network surrogate to learn the solution to our optimization problem. By running the `train_nn.py` script, we train our surrogate model, and store the weights as a .pt file in the /models folder.

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

## 3. What to Verify?

By leveraging the MILP engine, we can verify both constraint violations and the worst-case sub-optimality. With the crown engine, we can only verify the constraint violations.





