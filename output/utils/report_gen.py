import nbformat as nbf
import os
from datetime import datetime
import numpy as np

def generate_report(results, config, output_path="output/"):
    nb = nbf.v4.new_notebook()
    check_type = config['model_meta'].get('check', 'distance')
    engine_type = config['model_meta'].get('engine', 'milp')
    solver_type = config['model_meta'].get('solver', 'gurobi')
    problem_type = config['model_meta'].get('ptype', 'acopf')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cons_names = config.get('verification_spec', {}).get('constraints', {}).get('names', [])
    v_spec = config.get('verification_spec', {})
    
    # --- 1. Header Section ---
    title = "# ⚠️ Constraint Violation Report" if check_type == "constraint" else "# 🛡️ Sub-Optimality Analysis Report"
    nb['cells'].append(nbf.v4.new_markdown_cell(f"{title}\n**Generated on:** {timestamp}"))
    
    # --- 2. Methodology Explanation ---
    if problem_type == "acopf":
        methodology_md = (
            "## 📖 Methodology: Interval-Based Verification\n"
            "This report uses **alpha-CROWN**, McCormick relaxations and alpha-max beta-min relaxations to verify the grid safety.\n\n"
            "1. **Interval Bounds:** Unlike point-checks, this method calculates the **Global Max/Min** "
            "for every operational constraint across the entire input uncertainty range.\n"
            "2. **Certified Safety:** If a 'Max Violation' is 0.000000, it is a **mathematical proof** "
            "that the grid will never exceed that limit within the specified load ranges.\n"
            "3. **Feasibility Residual (Ibal):** Measures the KCL mismatch. High residuals indicate "
            "the Neural Network is struggling to maintain physical power flow consistency."
        )
    elif check_type == "constraint" and problem_type != "acopf":
        methodology_md = (
            "## 📖 Methodology & Interpretation\n"
            "This report provides a **formal safety audit** of the Neural Network. Please note the following regarding the results:\n\n"
            "1. **Worst-Case Search:** The counter-example shown below is the result of a Global Optimization (MILP) that identified the exact point in the input space "
            "where the Neural Network violates the physical constraints most severely.\n"
            "2. **Snapshot Inference:** All values in the tables below (Inputs, Outputs, and Violations) are derived from this **single worst-case point**. "
            "While one constraint may be the 'Max Violator' at this point, the other rows show how the rest of the system behaves at that same failing state. "
            "**Note:** There may exist other inputs where different constraints are violated more heavily than what is portrayed in this specific snapshot.\n"
            r"3. **Equality Relaxation:** Physical equalities (e.g., $A=B$) are mathematically unrolled into two inequalities ($A-B \le \epsilon$ and $B-A \le \epsilon$) "
            "using a precision tolerance of $10^{-6}$. This allows you to identify whether the failure is a **shortfall** or an **excess** relative to the target balance.\n"
            "4. **Global Guarantees:** If the 'Worst Violation' is within tolerance, it constitutes a **mathematical proof** that the network is safe across "
            "the *entire* continuous input range defined in the configuration."
        )
    else:
        methodology_md = (
            "## 📖 Methodology: Sub-Optimality Analysis\n"
            "This report audits the **economic efficiency** of the NN compared to the ground truth solution.\n\n"
            "1. **Max Sub-Optimality Search:** The verifier searches for the input state that produces the largest **Optimality Gap** compared to the ground truth. "
            "This is the 'worst-case' scenario for economic performance.\n"
            "2. **Bi-level Optimization:** For the identified worst-case input, we calculate:\n"
            "    - **NN Cost:** The objective value achieved using the NN's predictions.\n"
            "    - **True Optimal:** The theoretical minimum cost found by a global solver.\n"
            "3. **Sub-Optimality Gap:** The gap is defined as $Cost_{NN} - Cost_{Optimal}$. This value represents the maximum "
            "possible 'money left on the table' by using the AI proxy instead of a full solver.\n"
            "4. **Performance Guarantee:** If the Proven Max Sub-Optimality is near zero, the NN is **formally certified** to be an "
            "optimal proxy for the underlying system across the defined search space."
        )
    nb['cells'].append(nbf.v4.new_markdown_cell(methodology_md))
    
    # --- 3. Metadata ---
    input_indices = v_spec.get('indices', {}).get('input_indices', [])
    output_indices = v_spec.get('indices', {}).get('output_indices', [])
    bounds_list = v_spec.get('input_bounds', []) 

    # --- 4. High-Level Metrics & Status ---
    solve_time = results.get('runtime_sec', 0.0)
    
    # --- 3. ACOPF Specific Summary Table ---
    if problem_type == "acopf":
        # 1. Overall Safety Calculation
        max_phys = max(results.get('Pg tot Max Violation', 0.0), results.get('Qg tot Max Violation', 0.0),
                       results.get('Vm tot Max Violation', 0.0), results.get('Ibr tot Max Violation', 0.0))
        status = "✅ CERTIFIED SAFE" if max_phys < 1e-6 else "❌ VIOLATION"
        
        # 2. Main Summary Cell
        summary_md = (
            f"### ⏱️ Performance & Safety Status\n"
            f"- **System Status:** `{status}`\n"
            f"- **Solve Time:** `{solve_time:.4f}s`\n"
            f"- **Engine:** `alpha-beta-CROWN`\n\n"
            f"| Physical Limit Group | Max Worst-Case Violation | Status |\n"
            f"| :--- | :--- | :--- |\n"
            f"| **Real Power (Pg)** | `{results.get('Pg tot Max Violation', 0.0):.6f}` | {'✅' if results.get('Pg tot Max Violation', 0.0) < 1e-6 else '❌'} |\n"
            f"| **Reactive Power (Qg)** | `{results.get('Qg tot Max Violation', 0.0):.6f}` | {'✅' if results.get('Qg tot Max Violation', 0.0) < 1e-6 else '❌'} |\n"
            f"| **Voltage Mag (Vm)** | `{results.get('Vm tot Max Violation', 0.0):.6f}` | {'✅' if results.get('Vm tot Max Violation', 0.0) < 1e-6 else '❌'} |\n"
            f"| **Branch Current (Ibr)** | `{results.get('Ibr tot Max Violation', 0.0):.6f}` | {'✅' if results.get('Ibr tot Max Violation', 0.0) < 1e-6 else '❌'} |"
        )
        nb['cells'].append(nbf.v4.new_markdown_cell(summary_md))
        
        # 3. KCL Residuals Cell
        ibal_md = (
            "### ⚖️ KCL / Feasibility Residuals\n"
            f"| Metric | Value | Interpretation |\n"
            f"| :--- | :--- | :--- |\n"
            f"| **Max I-Balance Residual** | `{results.get('Ibal tot Max Violation', 0.0):.6f}` | Worst-case bus mismatch |\n"
            f"| **Avg I-Balance Residual** | `{results.get('Ibal tot Avg Violation', 0.0):.6f}` | System-wide mean mismatch |"
        )
        nb['cells'].append(nbf.v4.new_markdown_cell(ibal_md))

        # 4. Detailed Directional Table Cell
        details_md = "### 📏 Detailed Directional Violations\n"
        details_md += "| Constraint | Max Violation | Avg Violation |\n| :--- | :--- | :--- |\n"
        for label, prefix in [('Pg Upper', 'Pg Up'), ('Pg Lower', 'Pg Down'), ('Qg Upper', 'Qg Up'), 
                             ('Qg Lower', 'Qg Down'), ('Vm Upper', 'Vm Up'), ('Vm Lower', 'Vm Down')]:
            m_v = results.get(f'{prefix} Max Violation', 0.0)
            a_v = results.get(f'{prefix} Avg Violation', 0.0)
            details_md += f"| {label} | `{m_v:.6f}` | `{a_v:.6f}` |\n"
        nb['cells'].append(nbf.v4.new_markdown_cell(details_md))


    elif check_type == "distance":
        gap = results.get('optimality_gap', 0.0)
        status = "✅ OPTIMAL" if gap < 1e-4 else "❌ SUBOPTIMAL"
        # Calculate costs
        c = np.array(v_spec.get('objective_c', []))
        nn_vals = np.array(results.get('full_x_vector', []))
        opt_vals = np.array(results.get('x_star', []))
        nn_cost = np.dot(c, nn_vals) if len(c) == len(nn_vals) else 0.0
        true_cost = np.dot(c, opt_vals) if len(c) == len(opt_vals) else 0.0

        audit_md = (
            f"### 📊 Optimality Summary\n"
            f"- **Check:** `{check_type}`\n"
            f"- **Status:** `{status}`\n"
            f"- **Proven Sub-optimality (Gap):** `{gap:.6f}`\n"
            f"- **NN System Cost:** `{nn_cost:.4f}` | **True Optimal Cost:** `{true_cost:.4f}`\n"
            f"- **Solve Time:** `{solve_time:.4f}s`\n"
            f"- **Engine:** `{engine_type}`"
        )
        nb['cells'].append(nbf.v4.new_markdown_cell(audit_md))
    else:
        max_viol = results.get('max_violation', 0.0)
        status = "✅ FEASIBLE" if max_viol < 1e-4 else "❌ VIOLATED"
        audit_md = (
            f"### ⏱️ Performance & Safety Result\n"
            f"- **Check:** `{check_type}`\n"
            f"- **Status:** `{status}`\n"
            f"- **Max Violation Found:** `{max_viol:.6f}`\n"
            f"- **Solve Time:** `{solve_time:.4f}s`\n"
            f"- **Engine:** `{engine_type}`\n"
            f"- **Solver:** `{solver_type}`"
        )
        nb['cells'].append(nbf.v4.new_markdown_cell(audit_md))
    
    if problem_type != "acopf":
        # ---   Search Space ---
        metadata_md = "### 📑 Verification Search Space\n| Variable | Search Range | Type |\n| :--- | :--- | :--- |\n"
        for i, idx in enumerate(input_indices):
            b = bounds_list[i] if i < len(bounds_list) else {'min': '?', 'max': '?'}
            metadata_md += f"| Input {idx} | `[{b.get('min')}, {b.get('max')}]` | Continuous |\n"
        nb['cells'].append(nbf.v4.new_markdown_cell(metadata_md))

    # --- 5. Variable Comparison (Distance Mode Only) ---
    if check_type == "distance":
        comp_md = "### ⚖️ Output Variable Comparison (The 'Why')\n"
        comp_md += "| Index | Cost Coeff | NN Value | True Opt | Diff |\n| :--- | :--- | :--- | :--- | :--- |\n"
        c = v_spec.get('objective_c', [])
        x_nn = results.get('full_x_vector', [])
        x_star = results.get('x_star', [])
        for i in output_indices:
            diff = x_nn[i] - x_star[i]
            comp_md += f"| {i} | {c[i]:.4f} | {x_nn[i]:.4f} | {x_star[i]:.4f} | {diff:+.4f} |\n"
        nb['cells'].append(nbf.v4.new_markdown_cell(comp_md))

    if problem_type != "acopf":
        # --- 6. Component Analysis Section ---
        in_vals = [f"{v:.4f}" for v in results.get('at_input_val', [])]
        out_vals = [f"{v:.4f}" for v in results.get('nn_vals', [])]
        comp_md = (
            "### 🔍 Component Analysis (Worst-Case Snapshot)\n"
            "| Component | Indices | Values |\n| :--- | :--- | :--- |\n"
            f"| **NN Inputs** | `{input_indices}` | `{in_vals}` |\n"
            f"| **NN Outputs** | `{output_indices}` | `{out_vals}` |"
        )
        nb['cells'].append(nbf.v4.new_markdown_cell(comp_md))

    # --- 7. Detailed Constraint Table (Constraint Mode Only) ---
    if check_type == "constraint" and problem_type != "acopf":
        worst_idx = results.get('worst_row_idx', -1)
        A = np.array(v_spec.get('constraints', {}).get('A', []))
        b = np.array(v_spec.get('constraints', {}).get('b_static', []))
        x_full = np.array(results.get('full_x_vector', []))
        
        table_md = "### 📏 Detailed Constraint Analysis\n"
        table_md += "| Row | Name | Status | LHS | Limit (b) | Violation |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n"
        
        for i in range(len(A)):
            lhs_val = np.dot(A[i], x_full)
            viol = lhs_val - b[i]
            is_worst = (i == worst_idx and viol > 1e-6)
            status_icon = "🔥 **MAX**" if is_worst else ("❌" if viol > 1e-6 else "✅")
            c_name = cons_names[i] if i < len(cons_names) else f"Row {i}"
            table_md += f"| {i} | {c_name} | {status_icon} | {lhs_val:.4f} | {b[i]:.4f} | {viol:.6f} |\n"
            
            if viol > 1e-6:
                contribs = [(j, A[i][j] * x_full[j]) for j in range(len(x_full)) if abs(A[i][j] * x_full[j]) > 1e-4]
                calc_str = " + ".join([f"({v:.2f}*x{j})" for j, v in contribs])
                table_md += f"| | ↳ *Calc:* | | `{calc_str}` | = | `{lhs_val:.4f}` |\n"
        nb['cells'].append(nbf.v4.new_markdown_cell(table_md))

    # --- 8. Visualizations ---
    if problem_type == "acopf":
        viz_code = f"""
import matplotlib.pyplot as plt
metrics = {{
    'Pg': {results.get('Pg tot Max Violation', 0.0)},
    'Qg': {results.get('Qg tot Max Violation', 0.0)},
    'Vm': {results.get('Vm tot Max Violation', 0.0)},
    'Ibr': {results.get('Ibr tot Max Violation', 0.0)}
}}
plt.figure(figsize=(8, 4))
plt.bar(metrics.keys(), metrics.values(), color=['#e74c3c' if v > 1e-6 else '#2ecc71' for v in metrics.values()])
plt.axhline(y=1e-6, color='gray', linestyle='--', label='Threshold')
plt.ylabel('Max Violation (pu)'); plt.title('ACOPF Safety Violations'); plt.legend(); plt.show()
        """
    
    elif check_type == "distance":
        viz_code = f"""
import matplotlib.pyplot as plt
import numpy as np
nn_vals = {[results.get('full_x_vector', [])[i] for i in output_indices]}
opt_vals = {[results.get('x_star', [])[i] for i in output_indices]}
indices = np.arange(len(nn_vals))
plt.figure(figsize=(10, 4))
plt.bar(indices-0.2, nn_vals, width=0.4, label='NN', color='#3498db')
plt.bar(indices+0.2, opt_vals, width=0.4, label='True Opt', color='#e74c3c')
plt.title('Output Variable Comparison'); plt.legend(); plt.show()
        """
    else:
        plot_names = [cons_names[i] if i < len(cons_names) else f"Row {i}" for i in range(len(A))]
        viz_code = f"""
import matplotlib.pyplot as plt
violations = {results.get('violation_per_row', [])}
plt.figure(figsize=(10, 4))
plt.barh({plot_names}, violations, color=['#e74c3c' if v > 1e-6 else '#2ecc71' for v in violations])
plt.axvline(x=0, color='black', lw=1); plt.title('Constraint Violations'); plt.show()
        """
    nb['cells'].append(nbf.v4.new_code_cell(viz_code.strip()))

    # --- 9. Save Logic ---
    filename = f"report_{check_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb"
    os.makedirs(output_path, exist_ok=True)
    full_path = os.path.join(output_path, filename)
    with open(full_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    return full_path