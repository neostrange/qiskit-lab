import numpy as np

# Core optimization module
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.problems import Variable, QuadraticObjective # Keep Variable for vartype

# Quantum algorithms and optimizers
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler 

# The wrapper for solving optimization problems with quantum algorithms
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import warnings

# Suppress specific Qiskit warnings that might be noisy in notebooks
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="The qiskit.algorithms.optimizers.COBYLA class is deprecated")


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
PENALTY_FACTOR = 1000 
print(f"INFO: PENALTY_FACTOR set to {PENALTY_FACTOR}")

# --- Classical Solver (A bit more complex than before, but still exhaustive) ---
print("\n--- Classical Optimization (Exhaustive Search for 4 Binary Variables) ---")

best_cost_classical = float('inf')
best_schedule_classical = None

def calculate_full_cost_and_penalties(x_chg_vals, x_dis_vals, initial_charge, penalty_factor, verbose=False):
    total_cost = 0.0
    current_battery_state = initial_charge
    current_penalty_sum = 0.0 

    if verbose: print(f"Initial Battery Charge: {initial_charge} kWh")
    if verbose: print(f"--- Tracing Classical Solution with Penalty Factor {penalty_factor} ---")

    for t in range(NUM_SLOTS):
        x_chg_t = x_chg_vals[t]
        x_dis_t = x_dis_vals[t]

        if x_chg_t == 1 and x_dis_t == 1:
            current_penalty_sum += penalty_factor 
            if verbose: print(f"Slot {t}: !!! PENALTY: Simultaneous C/D - Penalty Added: {penalty_factor}")

        net_local_energy = SOLAR[t] - LOAD[t]
        battery_exchange = x_dis_t * BATTERY_MAX_EXCHANGE - x_chg_t * BATTERY_MAX_EXCHANGE

        next_battery_state = current_battery_state + battery_exchange

        if next_battery_state > BATTERY_CAPACITY:
            penalty_amount = penalty_factor * (next_battery_state - BATTERY_CAPACITY)
            current_penalty_sum += penalty_amount
            if verbose: print(f"Slot {t}: !!! PENALTY: Overcharge ({next_battery_state:.1f} kWh > {BATTERY_CAPACITY} kWh) - Added: {penalty_amount:.2f}")

        if next_battery_state < 0:
            penalty_amount = penalty_factor * (-next_battery_state)
            current_penalty_sum += penalty_amount
            if verbose: print(f"Slot {t}: !!! PENALTY: Undercharge ({next_battery_state:.1f} kWh < 0 kWh) - Added: {penalty_amount:.2f}")

        net_energy_after_battery = net_local_energy + battery_exchange

        cost_this_slot = 0.0
        if net_energy_after_battery > 0: 
            cost_this_slot -= net_energy_after_battery * SELL_PRICE[t]
        elif net_energy_after_battery < 0: 
            cost_this_slot += -net_energy_after_battery * BUY_PRICE[t]
        
        total_cost += cost_this_slot
        current_battery_state = next_battery_state 

        if verbose:
            print(f"Slot {t}: Chg={x_chg_t}, Dis={x_dis_t}")
            print(f"  Local Net: {net_local_energy:.1f} kWh, Bat Exchange: {battery_exchange:.1f} kWh")
            print(f"  Prev Bat State: {current_battery_state-battery_exchange:.1f} kWh -> Next Bat State: {current_battery_state:.1f} kWh, Net after Bat: {net_energy_after_battery:.1f} kWh")
            print(f"  Slot Cost: ${cost_this_slot:.2f}, Current Penalties Sum: ${current_penalty_sum:.2f}")

    if verbose: print(f"--- End of Classical Solution Trace ---")
    return total_cost + current_penalty_sum 

print("INFO: Performing exhaustive classical search...")
for i in range(2** (NUM_SLOTS * 2)): 
    schedule_bits = []
    temp_i = i
    for _ in range(NUM_SLOTS * 2):
        schedule_bits.append(temp_i % 2)
        temp_i //= 2
    schedule_bits.reverse() 

    x_chg = [schedule_bits[0], schedule_bits[2]] 
    x_dis = [schedule_bits[1], schedule_bits[3]] 

    current_full_cost = calculate_full_cost_and_penalties(x_chg, x_dis, BATTERY_INITIAL_CHARGE, PENALTY_FACTOR)

    if current_full_cost < best_cost_classical:
        best_cost_classical = current_full_cost
        best_schedule_classical = {
            'x_chg': x_chg,
            'x_dis': x_dis
        }

print(f"\nOptimal Classical Solution (x_chg, x_dis for each slot): {best_schedule_classical}")
print(f"Minimum Cost (incl. penalties): ${best_cost_classical:.2f}")
print("-" * 30)


# --- Quantum Optimization (QAOA on Simulator) ---
print("\n--- Quantum Optimization (QAOA on Simulator) ---")

qp = QuadraticProgram("battery_optimization")
print("DEBUG: QuadraticProgram created.")

x_chg_vars = [qp.binary_var(name=f'x_chg_{t}') for t in range(NUM_SLOTS)]
x_dis_vars = [qp.binary_var(name=f'x_dis_{t}') for t in range(NUM_SLOTS)]
print(f"INFO: Defined {qp.get_num_vars()} binary variables: {', '.join([v.name for v in qp.variables])}")

_objective_constant = 0.0 
_objective_linear = {}
_objective_quadratic = {}
print("DEBUG: Internal objective coefficient dictionaries initialized.")

# --- 1. Penalty for simultaneous charge and discharge ---
print("\n--- Building Objective: Simultaneous Charge/Discharge Penalties ---")
for t in range(NUM_SLOTS):
    key = tuple(sorted((x_chg_vars[t], x_dis_vars[t]), key=lambda x: x.name))
    _objective_quadratic[key] = _objective_quadratic.get(key, 0.0) + PENALTY_FACTOR
    print(f"INFO: Added {PENALTY_FACTOR} * {x_chg_vars[t].name} * {x_dis_vars[t].name} for simultaneous C/D penalty.")
print("DEBUG: Simultaneous C/D penalties added.")

# --- 2. Cost/Benefit based on grid interaction ---
print("\n--- Building Objective: Grid Interaction Costs/Benefits ---")
for t in range(NUM_SLOTS):
    charge_cost = BUY_PRICE[t] * BATTERY_MAX_EXCHANGE
    _objective_linear[x_chg_vars[t]] = _objective_linear.get(x_chg_vars[t], 0.0) + charge_cost
    print(f"INFO: Added {charge_cost:.2f} * {x_chg_vars[t].name} for charging cost.")

    discharge_benefit = -SELL_PRICE[t] * BATTERY_MAX_EXCHANGE
    _objective_linear[x_dis_vars[t]] = _objective_linear.get(x_dis_vars[t], 0.0) + discharge_benefit
    print(f"INFO: Added {discharge_benefit:.2f} * {x_dis_vars[t].name} for discharging benefit.")
print("DEBUG: Grid interaction costs/benefits added.")

# --- 3. Battery State Tracking and Penalties for Capacity (Manual Expansion) ---
print("\n--- Building Objective: Battery State & Capacity Penalties (Manual Expansion) ---")

S_t_current_constant = float(BATTERY_INITIAL_CHARGE) 
S_t_current_linear = {}
S_t_current_quadratic = {} 

print("DEBUG: Initial battery state at t=0 before any exchange:")
print(f"  S_t_current_constant = {S_t_current_constant}")
print(f"  S_t_current_linear = {S_t_current_linear}")
print(f"  S_t_current_quadratic = {S_t_current_quadratic}")

for t in range(NUM_SLOTS):
    print(f"\n--- Time Step t={t} ---")
    print(f"INFO: Battery state S_{t} (before this step's exchange):")
    print(f"  Constant: {S_t_current_constant:.2f}")
    print(f"  Linear Terms: {S_t_current_linear}")
    
    term_dis_coeff = float(BATTERY_MAX_EXCHANGE) 
    term_chg_coeff = float(-BATTERY_MAX_EXCHANGE) 

    print(f"INFO: Exchange terms for this step:")
    print(f"  Discharge term: {term_dis_coeff:.2f} * {x_dis_vars[t].name}")
    print(f"  Charge term: {term_chg_coeff:.2f} * {x_chg_vars[t].name}")

    S_t_next_constant = S_t_current_constant
    S_t_next_linear = S_t_current_linear.copy() 
    S_t_next_quadratic = S_t_current_quadratic.copy() 

    S_t_next_linear[x_dis_vars[t]] = S_t_next_linear.get(x_dis_vars[t], 0.0) + term_dis_coeff
    S_t_next_linear[x_chg_vars[t]] = S_t_next_linear.get(x_chg_vars[t], 0.0) + term_chg_coeff

    print(f"INFO: Battery state S_{t+1} (after this step's exchange):")
    print(f"  Constant: {S_t_next_constant:.2f}")
    print(f"  Linear Terms: {S_t_next_linear}")
    
    # Penalty 1: PENALTY_FACTOR * (S_t_next - BATTERY_CAPACITY)**2
    print(f"INFO: Calculating OVER-CAPACITY Penalty (P * (S_{t+1} - C_cap)^2) with P={PENALTY_FACTOR}")
    
    penalty_over_capacity_const_contribution = (S_t_next_constant - BATTERY_CAPACITY)**2
    _objective_constant += (penalty_over_capacity_const_contribution * PENALTY_FACTOR)
    print(f"  Added constant {penalty_over_capacity_const_contribution:.2f} * {PENALTY_FACTOR} to objective constant.")
    
    for var, coeff in S_t_next_linear.items():
        linear_coeff_to_add = 2 * (S_t_next_constant - BATTERY_CAPACITY) * coeff * PENALTY_FACTOR
        _objective_linear[var] = _objective_linear.get(var, 0.0) + linear_coeff_to_add
        print(f"  Added linear term {linear_coeff_to_add:.2f} * {var.name} to objective.")

    linear_vars_list = list(S_t_next_linear.keys())
    for i in range(len(linear_vars_list)):
        v_i = linear_vars_list[i]
        c_i = S_t_next_linear[v_i]

        key = tuple(sorted((v_i, v_i), key=lambda x: x.name))
        quadratic_coeff_to_add = c_i**2 * PENALTY_FACTOR
        _objective_quadratic[key] = _objective_quadratic.get(key, 0.0) + quadratic_coeff_to_add
        print(f"  Added quadratic term {quadratic_coeff_to_add:.2f} * {v_i.name}^2 to objective.")

        for j in range(i + 1, len(linear_vars_list)):
            v_j = linear_vars_list[j]
            c_j = S_t_next_linear[v_j]
            key = tuple(sorted((v_i, v_j), key=lambda x: x.name))
            quadratic_coeff_to_add = 2 * c_i * c_j * PENALTY_FACTOR
            _objective_quadratic[key] = _objective_quadratic.get(key, 0.0) + quadratic_coeff_to_add
            print(f"  Added quadratic term {quadratic_coeff_to_add:.2f} * {v_i.name} * {v_j.name} to objective.")


    # Penalty 2: PENALTY_FACTOR * S_t_next**2 (for non-negativity)
    print(f"INFO: Calculating UNDER-CAPACITY (Non-Negativity) Penalty (P * S_{t+1}^2) with P={PENALTY_FACTOR}")
    
    penalty_under_capacity_const_contribution = S_t_next_constant**2
    _objective_constant += (penalty_under_capacity_const_contribution * PENALTY_FACTOR)
    print(f"  Added constant {penalty_under_capacity_const_contribution:.2f} * {PENALTY_FACTOR} to objective constant.")

    for var, coeff in S_t_next_linear.items():
        linear_coeff_to_add = 2 * S_t_next_constant * coeff * PENALTY_FACTOR
        _objective_linear[var] = _objective_linear.get(var, 0.0) + linear_coeff_to_add
        print(f"  Added linear term {linear_coeff_to_add:.2f} * {var.name} to objective.")

    for i in range(len(linear_vars_list)):
        v_i = linear_vars_list[i]
        c_i = S_t_next_linear[v_i]

        key = tuple(sorted((v_i, v_i), key=lambda x: x.name))
        quadratic_coeff_to_add = c_i**2 * PENALTY_FACTOR
        _objective_quadratic[key] = _objective_quadratic.get(key, 0.0) + quadratic_coeff_to_add
        print(f"  Added quadratic term {quadratic_coeff_to_add:.2f} * {v_i.name}^2 to objective.")

        for j in range(i + 1, len(linear_vars_list)):
            v_j = linear_vars_list[j]
            c_j = S_t_next_linear[v_j]
            key = tuple(sorted((v_i, v_j), key=lambda x: x.name))
            quadratic_coeff_to_add = 2 * c_i * c_j * PENALTY_FACTOR
            _objective_quadratic[key] = _objective_quadratic.get(key, 0.0) + quadratic_coeff_to_add
            print(f"  Added quadratic term {quadratic_coeff_to_add:.2f} * {v_i.name} * {v_j.name} to objective.")
    
    print(f"\n--- Cumulative Objective Terms (Internal) after Time Step t={t} ---")
    print(f"  Constant: {_objective_constant:.2f}")
    print(f"  Linear: {_objective_linear}")
    print(f"  Quadratic: {_objective_quadratic}")
    print("--------------------------------------------------------")

    S_t_current_constant = S_t_next_constant
    S_t_current_linear = S_t_next_linear
    S_t_current_quadratic = S_t_next_quadratic 

print("DEBUG: All battery capacity and non-negativity penalties added to internal coefficients.")


# --- Set the QP Objective from Collected Coefficients ---
print("\n--- Setting QuadraticProgram Objective ---")

# Convert Qiskit Variable objects in _objective_linear and _objective_quadratic to strings (their names)
# as qp.minimize expects string keys for its linear and quadratic dictionaries.
final_linear_coeffs = {var.name: coeff for var, coeff in _objective_linear.items()}

# Ensure quadratic keys are tuples of strings (variable names)
final_quadratic_coeffs = {}
for (var1, var2), coeff in _objective_quadratic.items():
    # Make sure the key is sorted by name for consistency, though it should be already if tuples are sorted
    key_as_names = tuple(sorted((var1.name, var2.name)))
    final_quadratic_coeffs[key_as_names] = coeff

qp.minimize(constant=float(_objective_constant),
            linear=final_linear_coeffs,
            quadratic=final_quadratic_coeffs)

print("INFO: QuadraticProgram objective set using `qp.minimize()`.")

print(f"\nINFO: Final QuadraticProgram Objective Function (After setting via qp.minimize()):")
print(f"  Constant: {qp.objective.constant}")
print(f"  Linear Terms: {qp.objective.linear.to_dict()}")
print(f"  Quadratic Terms: {qp.objective.quadratic.to_dict()}")


# --- Convert to QUBO ---
print("\n--- QUBO Conversion ---")
converter = QuadraticProgramToQubo()
print("DEBUG: QuadraticProgramToQubo converter created.")
qubo = converter.convert(qp)
print(f"INFO: Problem converted to QUBO. Number of qubits: {qubo.get_num_vars()}")

# It's useful to inspect the final QUBO structure:
print(f"\nINFO: Final QUBO Objective Function (After conversion):")
print(f"  Constant: {qubo.objective.constant}")
print(f"  Linear Terms: {qubo.objective.linear.to_dict()}")
print(f"  Quadratic Terms: {qubo.objective.quadratic.to_dict()}")


# --- Run QAOA ---
print("\n--- Running QAOA ---")
sampler = Sampler()
print("DEBUG: Sampler created.")
qaoa = QAOA(sampler=sampler, optimizer=COBYLA(), reps=1) # reps=1 for quick test, consider increasing for better results
print(f"DEBUG: QAOA object created with {qaoa.reps} repetition(s) and {type(qaoa.optimizer).__name__} optimizer.")
optimizer = MinimumEigenOptimizer(qaoa)
print("DEBUG: MinimumEigenOptimizer created.")

print("INFO: Solving QUBO using QAOA...")
result = optimizer.solve(qubo)
print("DEBUG: Optimization result obtained.")

print("\n--- Quantum Optimization Result ---")
print(result.prettyprint())

# Extract the solution in terms of the original variables using variables_dict
x_chg_quantum = [int(result.variables_dict[f'x_chg_{t}']) for t in range(NUM_SLOTS)]
x_dis_quantum = [int(result.variables_dict[f'x_dis_{t}']) for t in range(NUM_SLOTS)]

# Verify the quantum solution using the classical cost function (which includes penalties)
print("\n--- Verifying Quantum Solution using Classical Cost Function ---")
quantum_solution_cost = calculate_full_cost_and_penalties(
    x_chg_quantum, x_dis_quantum, BATTERY_INITIAL_CHARGE, PENALTY_FACTOR, verbose=True
)

print(f"\nOptimal Quantum Schedule (x_chg, x_dis for each slot): {{'x_chg': {x_chg_quantum}, 'x_dis': {x_dis_quantum}}}")
print(f"Quantum Solution Total Cost (including penalties): ${quantum_solution_cost:.2f}")
print("-" * 30)

print("\n--- Comparison ---")
print(f"Classical Best Solution: {best_schedule_classical}, Cost: ${best_cost_classical:.2f}")
print(f"Quantum Found Solution: {{'x_chg': {x_chg_quantum}, 'x_dis': {x_dis_quantum}}}, Cost: ${quantum_solution_cost:.2f}")

# Check if the quantum solution matches the classical one (or is very close)
if abs(quantum_solution_cost - best_cost_classical) < 1e-6 and \
   x_chg_quantum == best_schedule_classical['x_chg'] and \
   x_dis_quantum == best_schedule_classical['x_dis']:
    print("\nSUCCESS: Quantum solution matches classical optimal solution (within tolerance)!")
else:
    print("\nWARNING: Quantum solution found does NOT perfectly match classical optimal solution.")
    print("This can happen due to QAOA being an approximate algorithm, optimizer limitations,")
    print("or the penalty terms in the QUBO not perfectly encoding the classical problem's constraints.")
    print("Consider:")
    print(f"  1. Increasing PENALTY_FACTOR (currently {PENALTY_FACTOR})")
    print(f"  2. Increasing QAOA 'reps' (currently {qaoa.reps})")
    print(f"  3. Trying a different classical optimizer for QAOA.")