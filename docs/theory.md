# 📖 The Theory of Neural Network Verification

Neural networks are non-linear, non-convex functions. Usually, we treat them as black boxes. However, for safety-critical applications, we need to guarantee that for any input $x$ within a range $[L, U]$, the output $y$ stays within a safe set $\mathcal{S}$.

This toolbox uses two primary methods to provide these guarantees: **MILP** and **CROWN**.

---

## 1. Exact Verification via MILP
To verify a network exactly, we must handle the non-linearity of the activation functions (like ReLU). 

A ReLU unit $y = \max(0, x)$ can be modeled exactly using **Big-M notation** and binary variables $z \in \{0, 1\}$. This transforms the Neural Network into a **Mixed-Integer Linear Programming (MILP)** problem:

1. If $z=1$, the neuron is active: $y = x$.
2. If $z=0$, the neuron is inactive: $y = 0$.

### The Verification Objective: Adversarial Search
Once the network is encoded as a set of linear constraints and binaries, the toolbox transforms the verification task into an **adversarial optimization problem**. Instead of testing random points, the solver performs a global search to maximize a **Violation Metric** ($V$):

$$\text{Maximize } V(x) = \text{Surrogate}(x) - \text{Physics\_Limit}(x)$$
$$\text{subject to: } x \in [L, U]$$

* **Safety Analysis (`constraint`):** The solver searches for the input $x$ that causes the largest physical violation. If the maximum $V \leq 0$, the model is formally proven safe.
* **Optimality Analysis (`distance`):** The solver maximizes the gap between the NN prediction and the true mathematical optimum. This identifies the "Worst-Case Error" across the entire domain.

By solving this to global optimality, the MILP engine provides a **Formal Certificate**: if the solver cannot find a single point where the metric exceeds your threshold, it is mathematically impossible for the network to fail within that domain.

### Pros & Cons
* **Precision:** If the solver (Gurobi/CPLEX) finds a solution, it is the mathematically "perfect" worst-case.
* **Counter-Examples:** If a safety property is broken, MILP gives you the exact input coordinates of the failure.
* **Cost:** It is computationally "NP-Hard." For very large networks, the number of binary combinations ($2^{neurons}$) becomes too large to solve quickly.

---

## 2. Bound-Based Verification via CROWN
CROWN (Linear Bound Propagation) takes a different approach. Instead of using binary variables, it "sandwiches" every neuron between two linear functions.

For any non-linear activation $\sigma(x)$, CROWN finds two lines such that:
$$\mathbf{A}_{low}x + b_{low} \leq \sigma(x) \leq \mathbf{A}_{up}x + b_{up}$$

By propagating these linear bounds from the input layer to the output layer, CROWN calculates a **Formal Bound** on the output.

### Pros & Cons
* **Speed:** It is significantly faster than MILP because it avoids "branching" (no binary variables).
* **Scalability:** It can handle much larger networks.
* **Conservativeness:** Because it uses approximations (the "sandwich"), the bounds it provides are wider than the truth. It might say a system is "potentially unsafe" when it is actually safe (a false positive), but it will **never** say a system is safe if it is actually unsafe.

---

## 🏗️ Summary Comparison

| Metric | MILP Engine | CROWN Engine |
| :--- | :--- | :--- |
| **Completeness** | **Complete** (Exact answer) | **Incomplete** (Conservative) |
| **Speed** | Slow (Exponential) | Fast (Polynomial) |
| **Output** | Exact Violation Point | Safety Certificate (Bounds) |
| **Best Use Case** | Final "Deep" Audit | Rapid Iterative Testing |