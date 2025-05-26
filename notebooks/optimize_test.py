import numpy as np

# Core optimization module (stable)
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

# Quantum algorithms and optimizers from qiskit-algorithms
# Note the change: 'qiskit.algorithms' -> 'qiskit_algorithms.minimum_eigensolvers'
from qiskit_algorithms.minimum_eigensolvers import QAOA
# Note the change: 'qiskit.algorithms.optimizers' -> 'qiskit_algorithms.optimizers'
from qiskit_algorithms.optimizers import COBYLA

# Primitives (stable)
from qiskit.primitives import Sampler

# The wrapper for solving optimization problems with quantum algorithms (stable)
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# --- 1. Define the Problem Parameters ---
costs = [3, 2, 1, 4] # C1, C2, C3, C4 for Slot 1, 2, 3, 4
num_slots = len(costs)

# --- 2. Classical Approach (for comparison) ---
print("--- Classical Optimization ---")
min_cost_classical = float('inf')
best_slot_classical = -1
for i in range(num_slots):
    if costs[i] < min_cost_classical:
        min_cost_classical = costs[i]
        best_slot_classical = i + 1 # Slot numbers start from 1
print(f"Optimal Classical Solution: Run appliance in Slot {best_slot_classical} for a cost of ${min_cost_classical}")
print("-" * 30)

# --- 3. Quantum Approach (QUBO Formulation in Qiskit) ---
print("--- Quantum Optimization (QAOA on Simulator) ---")

# Create a QuadraticProgram instance
qp = QuadraticProgram("appliance_scheduling")

# Add binary variables for each slot
for i in range(num_slots):
    qp.binary_var(name=f'x{i}')

# Set the objective function: minimize sum(cost_i * x_i)
linear_objective = {f'x{i}': costs[i] for i in range(num_slots)}
qp.minimize(linear=linear_objective)

# Add the "exactly one slot" constraint (x0 + x1 + x2 + x3 = 1)
sum_vars_linear = {f'x{i}': 1.0 for i in range(num_slots)} # Use float for coefficients
qp.linear_constraint(linear=sum_vars_linear, sense='==', rhs=1.0, name='one_slot_constraint')

# Convert the QuadraticProgram to a QUBO
qubo_converter = QuadraticProgramToQubo()
qubo = qubo_converter.convert(qp)

print("QUBO problem formulation (Qiskit's internal representation):")
print(f"Number of binary variables in QUBO: {qubo.get_num_binary_vars()}")
# print(f"QUBO objective and constraints:\n{qubo.export_as_str()}")
print("-" * 30)

# Define the quantum optimizer (QAOA)
# QAOA now takes 'reps' directly as an argument, no longer through 'p'
qaoa_mes = QAOA(sampler=Sampler(), optimizer=COBYLA(), reps=1)

# Use MinimumEigenOptimizer to solve the QUBO with QAOA
optimizer = MinimumEigenOptimizer(qaoa_mes)

# Solve the problem
qaoa_result = optimizer.solve(qubo)

print("QAOA Optimization Results:")
print(f"Optimal value found (Qiskit): {qaoa_result.fval}")
print(f"Binary solution (x0, x1, x2, x3): {qaoa_result.x}")

# Interpret the binary solution
chosen_slot_idx = -1
for i, val in enumerate(qaoa_result.x):
    if np.isclose(val, 1.0, atol=1e-5): # Use a small tolerance for float comparisons
        chosen_slot_idx = i
        break

if chosen_slot_idx != -1:
    print(f"Optimal Quantum Solution: Run appliance in Slot {chosen_slot_idx + 1} for a cost of ${costs[chosen_slot_idx]}")
else:
    print("No unique slot chosen by the quantum optimizer. This might happen due to noisy simulation or optimizer not converging perfectly for `reps=1`.")
    print("Consider increasing `reps` for QAOA or using a different classical optimizer for more robust results, especially for more complex problems.")

print("-" * 30)
print(f"Number of qubits required for this problem: {num_slots}")
print("Note: For current small problems on simulators, classical methods are significantly faster.")
print("The purpose here is to demonstrate the methodology of formulating and solving an optimization problem on a quantum computer, highlighting its potential for much larger and intractable problems in the future.")