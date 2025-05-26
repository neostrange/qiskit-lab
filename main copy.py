# Import classes from your defined modules
from models.energy_problem import EnergyManagementProblem
from optimizers.classical_optimizer import ClassicalOptimizer
from optimizers.quantum_optimizer import QuantumOptimizer
from analysis.solution_analyzer import SolutionAnalyzer

if __name__ == "__main__":
    # Define Problem Parameters
    params = {
        "num_slots": 2,
        "load": [5, 3],
        "solar": [4, 2],
        "buy_price": [0.20, 0.30],
        "sell_price": [0.10, 0.15],
        "battery_capacity": 3,
        "battery_initial_charge": 1,
        "battery_max_exchange": 1,
        "battery_efficiency": 1.0,
        "penalty_factor": 1000
    }
    print(f"INFO: PENALTY_FACTOR set to {params['penalty_factor']}")

    # 1. Instantiate the problem
    energy_problem = EnergyManagementProblem(**params)

    # 2. Run Classical Optimization
    classical_optimizer = ClassicalOptimizer(energy_problem)
    classical_schedule, classical_cost = classical_optimizer.optimize()

    # 3. Run Quantum Optimization
    qaoa_reps = 3 # Can be adjusted
    quantum_optimizer = QuantumOptimizer(energy_problem, qaoa_reps=qaoa_reps)
    quantum_schedule, quantum_qubo_val = quantum_optimizer.solve()

    # 4. Analyze Results
    analyzer = SolutionAnalyzer(energy_problem)
    analyzer.compare_results(classical_schedule, classical_cost, quantum_schedule, quantum_qubo_val)