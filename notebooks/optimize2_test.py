import numpy as np

# Core optimization module
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

# Quantum algorithms and optimizers
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler # V1 Sampler to avoid TypeError with qiskit-algorithms

# The wrapper for solving optimization problems with quantum algorithms
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# --- Problem Parameters ---
# Time slots
NUM_SLOTS = 2 # t=0, t=1

# Energy demands (kWh)
LOAD = [5, 3]
# Solar generation (kWh)
SOLAR = [4, 2]

# Grid prices ($/kWh)
BUY_PRICE = [0.20, 0.30]
SELL_PRICE = [0.10, 0.15]

# Battery parameters
BATTERY_CAPACITY = 3 # kWh
BATTERY_INITIAL_CHARGE = 1 # kWh
BATTERY_MAX_EXCHANGE = 1 # kWh per slot (simplified: if x_chg/x_dis=1, it moves 1 kWh)
BATTERY_EFFICIENCY = 1.0 # 100% for simplicity

# Penalty factor for constraints (needs to be large enough)
PENALTY_FACTOR = 100 # Adjust if solutions are invalid

# --- Classical Solver (A bit more complex than before, but still exhaustive) ---
# This will be a manual search over all 2^4 = 16 combinations
# for x_chg_0, x_dis_0, x_chg_1, x_dis_1
print("--- Classical Optimization (Exhaustive Search for 4 Binary Variables) ---")

best_cost_classical = float('inf')
best_schedule_classical = None

# Iterate through all 2^NUM_VARS combinations
for i in range(2** (NUM_SLOTS * 2)): # 2 binary vars per slot, NUM_SLOTS slots
    # Decode the binary combination into our variables
    # x_chg_0, x_dis_0, x_chg_1, x_dis_1
    schedule = []
    temp_i = i
    for _ in range(NUM_SLOTS * 2):
        schedule.append(temp_i % 2)
        temp_i //= 2
    schedule.reverse() # [x_chg_0, x_dis_0, x_chg_1, x_dis_1]

    x_chg = [schedule[0], schedule[2]] # x_chg_0, x_chg_1
    x_dis = [schedule[1], schedule[3]] # x_dis_0, x_dis_1

    current_cost = 0.0
    current_battery_state = BATTERY_INITIAL_CHARGE
    is_valid_schedule = True

    for t in range(NUM_SLOTS):
        # Constraint 1: Cannot simultaneously charge and discharge
        if x_chg[t] == 1 and x_dis[t] == 1:
            is_valid_schedule = False
            break

        # Calculate energy balance for this slot
        net_local_energy = SOLAR[t] - LOAD[t]

        battery_exchange = 0
        if x_chg[t] == 1:
            battery_exchange = -BATTERY_MAX_EXCHANGE # Energy drawn to charge battery
        elif x_dis[t] == 1:
            battery_exchange = BATTERY_MAX_EXCHANGE  # Energy provided by discharging battery

        # Update battery state (pre-check constraints)
        temp_battery_state = current_battery_state + battery_exchange

        # Constraint 2: Battery Capacity (overcharge/undercharge)
        if not (0 <= temp_battery_state <= BATTERY_CAPACITY):
            is_valid_schedule = False
            break
        
        # If valid so far, update battery state
        current_battery_state = temp_battery_state

        # Energy flow to/from grid
        net_energy_after_battery = net_local_energy + battery_exchange

        if net_energy_after_battery > 0: # Energy surplus, sell to grid
            current_cost -= net_energy_after_battery * SELL_PRICE[t]
        elif net_energy_after_battery < 0: # Energy deficit, buy from grid
            current_cost += -net_energy_after_battery * BUY_PRICE[t]
        # If net_energy_after_battery == 0, cost is 0 for grid interaction

    if is_valid_schedule:
        if current_cost < best_cost_classical:
            best_cost_classical = current_cost
            best_schedule_classical = {
                'x_chg': x_chg,
                'x_dis': x_dis
            }

print(f"Optimal Classical Solution (x_chg, x_dis for each slot): {best_schedule_classical}")
print(f"Minimum Cost: ${best_cost_classical:.2f}")
print("-" * 30)


# --- Quantum Approach (QUBO Formulation in Qiskit) ---
print("--- Quantum Optimization (QAOA on Simulator) ---")

qp = QuadraticProgram("battery_optimization")

# Binary variables: x_chg_0, x_dis_0, x_chg_1, x_dis_1
x_chg_vars = [qp.binary_var(name=f'x_chg_{t}') for t in range(NUM_SLOTS)]
x_dis_vars = [qp.binary_var(name=f'x_dis_{t}') for t in range(NUM_SLOTS)]

# Objective function and constraints need to be added using expressions of the binary variables
# This part is more involved as it combines linear and quadratic terms from penalties and costs

# A function to calculate cost from binary variables (for objective and penalties)
def calculate_cost_and_battery(x_chg_vals, x_dis_vals, initial_charge, penalty_factor):
    total_cost = 0.0
    current_battery_state = initial_charge
    penalty_sum = 0.0 # To track penalty for violated constraints

    for t in range(NUM_SLOTS):
        x_chg_t = x_chg_vals[t]
        x_dis_t = x_dis_vals[t]

        # Constraint 1: Cannot simultaneously charge and discharge (x_chg_t + x_dis_t <= 1)
        # We add (x_chg_t + x_dis_t - 1)^2 if it exceeds 1
        # For binary variables, if x_chg_t = x_dis_t = 1, then (1+1-1)^2 = 1. Else 0.
        penalty_sum += PENALTY_FACTOR * (x_chg_t + x_dis_t - 1)**2

        # Energy balance for this slot
        net_local_energy = SOLAR[t] - LOAD[t]
        battery_exchange = x_dis_t * BATTERY_MAX_EXCHANGE - x_chg_t * BATTERY_MAX_EXCHANGE

        # Update battery state (for constraints)
        next_battery_state = current_battery_state + battery_exchange

        # Constraint 2: Battery Capacity (next_battery_state <= BATTERY_CAPACITY)
        # Using a soft penalty for now: if it exceeds, penalize
        penalty_sum += PENALTY_FACTOR * max(0, next_battery_state - BATTERY_CAPACITY)**2

        # Constraint 3: Battery Non-negativity (next_battery_state >= 0)
        penalty_sum += PENALTY_FACTOR * max(0, -next_battery_state)**2

        # Grid interaction cost
        net_energy_after_battery = net_local_energy + battery_exchange
        if net_energy_after_battery >= 0: # Surplus, sell
            total_cost -= net_energy_after_battery * SELL_PRICE[t]
        else: # Deficit, buy
            total_cost += -net_energy_after_battery * BUY_PRICE[t]

        current_battery_state = next_battery_state

    return total_cost + penalty_sum

# Manually construct the objective and penalty terms for the QUBO
# This is complex because the energy balance depends on other variables across time steps
# For this specific problem, using Qiskit Optimization's 'quadratic_program_to_qubo'
# with `qp.quadratic_constraint` is the preferred way if the problem is defined as
# a QuadraticProgram.
# Let's try to define the constraints in qp and let the converter handle it.

# Define variables for battery state at end of each slot (B_0, B_1)
# Need to decide how to encode these if they are to be part of constraints in QP.
# For simplicity in QUBO, let's keep them as derived values for now
# and directly write the penalties from the binary variables.

# Construct the objective (linear + quadratic)
# This is the most complex part to do manually for a non-trivial problem.
# For demo, let's use the explicit constraint method in QP.

# 1. Objective: Minimize total cost (this needs to be expressed in terms of x_chg_t and x_dis_t)
# This objective requires careful substitution of E_buy, E_sell, E_chg, E_dis.
# Qiskit Optimization's `QuadraticProgram` is designed for this.

# Let's try to define the energy balance and cost in terms of QP expressions directly.
# This requires a bit of work to convert `max(0, X)` into quadratic form if not directly supported by QP.
# A common way to handle max(0, X) or min(0, X) in QUBO is by introducing auxiliary binary variables
# and enforcing the condition X > 0 or X < 0.

# Given the complexity of manually building QUBO for this, let's define as a QuadraticProgram
# and let the converter handle the quadratic constraint generation.

# Re-simplifying the problem for a demo-friendly QUBO (still more complex than appliance):
# We need variables for actual energy amounts, not just binary choices to charge/discharge fixed amounts.
# Let's try with 1-bit resolution for max values of 1, 2, 3, 4, 5kWh (requires 3 bits for 5kWh)
# This rapidly increases qubits: (4 variables * 3 bits/variable * 2 slots) = 24 qubits. Too much.

# Let's stick with the 4 binary variables (x_chg_0, x_dis_0, x_chg_1, x_dis_1)
# and simplify the objective function directly into QUBO, incorporating penalties.

# Objective for QUBO based on the paper example
# The objective function is usually defined with linear and quadratic terms (coeffs * var + coeffs * var1 * var2)
# The `calculate_cost_and_battery` function above computes the total cost for a given set of binary values.
# To convert this into a QUBO directly without `QuadraticProgram.linear_constraint`
# is highly manual and error-prone for non-trivial logic.

# The `QuadraticProgram` approach for complex problems:
# 1. Define all binary variables for actions.
# 2. Define auxiliary variables for energy flow or battery state if needed (and their bit representations).
# 3. Add linear constraints for energy balance (e.g., Solar + Buy + Disch = Load + Sell + Chg).
# 4. Add linear/quadratic constraints for battery capacity, simultaneous actions, etc.
# 5. Let `QuadraticProgramToQubo` convert these into a large QUBO.

# Given the feasibility report context, it's better to show the conceptual setup in Qiskit
# rather than manually deriving a large, error-prone QUBO.

# --- Let's set up the QP with linear constraints and let the converter do its work ---

# Define decision variables: binary for each slot, 1 if action happens, 0 otherwise
# x_chg_0, x_dis_0, x_chg_1, x_dis_1
# Additionally, to account for energy amounts, we could define:
# x_buy_amount_0, x_sell_amount_0, x_buy_amount_1, x_sell_amount_1
# These would be integers and would need bit encoding.
# This pushes us to too many qubits for practical QAOA sim.

# Let's revert to a simpler model that represents the "spirit" of energy management.
# A very basic energy scheduling with discrete options.

# Problem: Simplified Energy Allocation (3 Time Slots, 2 Energy Sources/Sinks)
# A homeowner needs to meet a fixed load in 3 time slots.
# They can use local generation OR buy from grid.
# The goal is to minimize buying cost.

# Variables:
# y_t: Binary (1 if use local generation in slot t, 0 if buy from grid in slot t)
# Cost of local generation: 0.0 (solar)
# Cost of buying: P_buy_t

# Total 3 qubits: y_0, y_1, y_2

# This is simply: minimize Sum( (1-y_t) * P_buy_t * Load_t )
# Which is: Sum(P_buy_t * Load_t) - Sum(y_t * P_buy_t * Load_t)
# Minimizing this is equivalent to maximizing Sum(y_t * P_buy_t * Load_t)
# So it's effectively "maximize using solar when it's expensive to buy".

# Let's add a constraint: can only use local generation if Solar_t >= Load_t
# If Solar_t < Load_t, then y_t must be 0 (must buy). This can be a penalty.

# This might be too simple, like the first problem. Let's stick with the battery problem
# but make the variables simpler to get a 4-qubit or 6-qubit system.

# Re-conceptualizing for 4-6 qubits:
# Variables:
# x_buy_0, x_sell_0, x_buy_1, x_sell_1 (binary: are we buying/selling a fixed unit, e.g., 1kWh)
# x_bat_0, x_bat_1 (binary: use battery or not, assume battery handles any fixed 1 unit)

# This will get very complex to form correctly for QAOA if actual energy balances are involved.

# **Recommendation for Feasibility Report:**

# Given the constraints and the complexity of QUBO formulation for multi-variable, multi-timestep problems with non-linear constraints (like battery min/max), the most impactful demonstration for a *feasibility report* is to:

# 1.  **Use the "Optimal Appliance Scheduling" example.** It's clean, clearly shows the QUBO concept, and yields a correct answer from the quantum side.
# 2.  **Focus the "Complexity Analysis" section on the *scaling* of problems that are *harder* than the demo.**
#     * **Classical Complexity:** Explain that for a real smart grid optimization with *N* houses, *M* devices per house, and *T* time slots, the number of interacting variables and complex constraints (`if-then`, `min/max`, non-linear costs) makes classical optimization (e.g., mixed-integer linear programming, non-linear programming) very computationally expensive, often resorting to heuristics. Mention NP-hard problems.
#     * **Quantum Complexity:** Explain that quantum computers encode these problems into Hamiltonians (like QUBOs). While current runtimes are slower due to simulators/noise, the *theoretical promise* for quantum algorithms (like QAOA for QUBOs, or Shor's for factoring, or Grover's for search) is that they can provide **super-polynomial speedups** (or polynomial for certain problems) over the best-known classical algorithms for specific problem classes.
#         * **Crucial Point:** Emphasize that the *bottleneck* for classical methods for these large-scale problems is the exponential growth of the search space, which quantum algorithms are designed to handle differently through superposition and entanglement.
#         * **Relate to Energy:** For complex resource allocation, smart grid optimization, or energy trading, where the number of parameters and choices explodes, quantum computers *may* find optimal or near-optimal solutions much faster than classical methods *in the future*.
#     * **Feasibility Conclusion:** Conclude that while practical quantum advantage for *large-scale* energy problems isn't here *today*, the theoretical foundations and early algorithmic successes (even on small problems) demonstrate the *feasibility* and *high potential* for quantum computing to tackle these problems in the future, justifying continued research and investment.

# **Why not a more complex direct demo for the report?**

# * **Qubit Count:** Most real-world energy problems (even simplified ones like the battery example with proper energy balance and multiple units) quickly exceed ~10-15 qubits when translated to QUBOs. Running QAOA on more than ~8-10 qubits reliably on a simulator (or current noisy hardware) for meaningful results is already challenging and time-consuming.
# * **QUBO Formulation Complexity:** Manually deriving the QUBO (all the $Q_{ij} x_i x_j$ terms) for even moderately complex problems is incredibly difficult and prone to errors. `qiskit_optimization`'s `QuadraticProgram` helps, but if you need to define very specific non-linear constraints or logic, it can still be tricky. For a report, focusing on *how* it's formulated for a simple case, and then *discussing* the scaling challenges, is more effective.

# **Therefore, I strongly recommend sticking with the "Optimal Appliance Scheduling" problem as your live demo/code example for the report.** Your text and explanation for the "complexity analysis" section will then cover the more advanced reasoning.

# ---

# **Revised Outline for Your Report (based on your tasks):**

# **1. Introduction**
#     * Briefly define optimization in the energy sector.
#     * Introduce the concept of quantum computing as a new paradigm for hard problems.
#     * State the report's purpose: analyze the feasibility of quantum computing for energy optimization by comparing classical and quantum approaches on a simple use case.

# **2. Problem Definition: Optimal Appliance Scheduling**
#     * Detail the problem: 4 time slots, varying costs, pick one.
#     * Show the table of costs.

# **3. Step 1: Classical Algorithm Performance & Complexity**
#     * **Algorithm:** Brute Force / Direct Inspection.
#     * **Demonstration:** Show your code snippet for the classical part and its output.
#     * **Performance (Speed):** Extremely fast for this small problem.
#     * **Complexity Analysis:**
#         * For `N` options, it's `O(N)` operations.
#         * Explain *why* this becomes intractable for larger, more realistic energy problems (e.g., choosing combinations of appliances, multi-time slot dispatch, or continuous energy amounts) where the search space grows exponentially. Mention NP-hard nature of real-world optimization problems.

# **4. Step 2: Quantum Algorithm Performance & Complexity**
#     * **Algorithm:** QAOA (Quantum Approximate Optimization Algorithm).
#     * **Quantum Formulation:**
#         * Explain QUBO: binary variables (qubits), objective function (cost), constraints as penalties.
#         * Show the structure of the QUBO (mention the 4 qubits).
#     * **Qiskit Implementation:**
#         * Highlight the `QuadraticProgram` for problem definition.
#         * Explain `QuadraticProgramToQubo` for converting to quantum-ready format.
#         * Show how `QAOA` (with `Sampler` and `COBYLA`) is used via `MinimumEigenOptimizer`.
#     * **Performance (Speed):**
#         * Show your quantum code output.
#         * **Crucial Discussion:** Explain that on a simulator, it's *slower* than classical. This is due to simulation overheads and the early stage of quantum hardware. Quantum computers are not a general-purpose speedup.
#     * **Complexity Analysis (Theoretical Potential vs. Current Reality):**
#         * **Theoretical:** QAOA aims for better scaling than classical heuristics for NP-hard problems, potentially achieving super-polynomial speedups or better approximate solutions. This is because quantum computers can explore the exponentially large solution space in parallel via superposition.
#         * **Current Reality:** Emphasize that current NISQ devices are limited by qubit count (4 qubits in our demo, but real problems need many more) and noise. This means practical quantum advantage for large-scale energy problems is still a future prospect.

# **5. Step 3: Comparative Results and Feasibility Analysis**
#     * **Comparison for Demo Problem:** Both methods yield the same correct answer (Slot 3, Cost $1). Classical is faster for this trivial case.
#     * **Feasibility Conclusion:**
#         * **Current Feasibility:** Currently, directly replacing classical algorithms for large-scale energy optimization problems with quantum algorithms on existing hardware is *not feasible* in terms of speed or accuracy.
#         * **Future Feasibility & Potential:** However, the **methodology is feasible and validated** on small scales. The theoretical advantages of quantum algorithms for NP-hard optimization problems (which are abundant in the energy sector) strongly indicate that as quantum hardware matures (more qubits, lower noise, error correction), quantum computing **will become feasible** and potentially outperform classical methods for truly challenging, intractable energy optimization problems.
#         * **Impact on Energy Sector:** Reiterate potential impacts (smart grid, renewable integration, market optimization).
#         * **Recommendation:** Continued research and development in quantum algorithms and hardware for the energy sector is a worthwhile investment, as quantum computing represents a paradigm shift for solving complex optimization problems.

# ---

# This revised structure allows you to fulfill all your tasks rigorously, leveraging your simple working code while accurately addressing the nuances of quantum computational power for a feasibility report.