"""
================================================================================
SolutionAnalyzer: Result Comparison for Battery Energy Management Optimization
================================================================================

This module provides the SolutionAnalyzer class, which compares and presents
results from classical and quantum optimizers for the battery energy management
problem.

Domain Context:
---------------
- Evaluates and contrasts solutions found by classical and quantum (QAOA)
  optimization methods.
- Reports cost, constraint satisfaction, and highlights differences or matches.

================================================================================
"""

# Import necessary classes from other modules
from models.energy_problem import EnergyManagementProblem

class SolutionAnalyzer:
    """
    Compares and presents results from classical and quantum optimizers.
    """
    def __init__(self, problem: EnergyManagementProblem):
        self.problem = problem

    def compare_results(self, classical_schedule, classical_cost, quantum_schedule, quantum_qubo_val):
        print("\n--- Comparison ---")
        quantum_classical_cost = self.problem.calculate_full_cost_and_penalties(
            quantum_schedule['x_chg'], quantum_schedule['x_dis'], verbose=True
        )

        print(f"\nClassical Best Solution: {classical_schedule}, Cost: ${classical_cost:.2f}")
        print(f"Quantum Found Solution: {{'x_chg': {quantum_schedule['x_chg']}, 'x_dis': {quantum_schedule['x_dis']}}}, Cost: ${quantum_classical_cost:.2f}")

        if abs(quantum_classical_cost - classical_cost) < 1e-6 and \
           quantum_schedule['x_chg'] == classical_schedule['x_chg'] and \
           quantum_schedule['x_dis'] == classical_schedule['x_dis']:
            print("\nSUCCESS: Quantum solution matches classical optimal solution (within tolerance)!")
        else:
            print("\nWARNING: Quantum solution found does NOT perfectly match classical optimal solution.")
            print("This can happen due to QAOA being an approximate algorithm, optimizer limitations,")
            print("or the penalty terms in the QUBO not perfectly encoding the classical problem's constraints.")
            print("Consider:")
            print(f"  1. Increasing PENALTY_FACTOR (currently {self.problem.penalty_factor})")
            print(f"  2. Increasing QAOA 'reps' (currently {self.problem.num_slots * 2 if self.problem.num_slots > 0 else 3})")
            print(f"  3. Trying a different classical optimizer for QAOA.")