"""
================================================================================
QuantumOptimizer: Leveraging QAOA for Microgrid Battery Optimization
================================================================================

This module provides the QuantumOptimizer class, designed to solve the
EnergyManagementProblem by formulating it as a Quadratic Unconstrained
Binary Optimization (QUBO) problem. It then utilizes Qiskit's Quantum
Approximate Optimization Algorithm (QAOA) to find approximate solutions
on a quantum simulator or quantum hardware.

The class handles the conversion of the energy management problem's costs
and constraints into a QUBO matrix, which is the native input format for
many quantum annealing and gate-model optimization algorithms.

Key functionalities include:
- **QUBO Construction:** Translates classical problem variables and objectives
  into binary variables and a quadratic objective function. This involves
  encoding costs (e.g., buying from grid) and penalties (e.g., battery
  violations) into the QUBO coefficients.
- **QAOA Integration:** Configures and runs QAOA using Qiskit, specifying
  the number of QAOA repetitions (p) and the classical optimizer.
- **Result Interpretation:** Converts the binary solutions from QAOA back
  into the physical schedule (charge/discharge decisions) and calculates
  its true cost using the `EnergyManagementProblem`'s verification method.

This optimizer serves as a proof-of-concept for applying quantum algorithms
to real-world energy management challenges, comparing their performance
against classical brute-force methods.

================================================================================
"""

import logging
import numpy as np
from qiskit_optimization.problems import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer # Correct wrapper
from qiskit_algorithms.minimum_eigensolvers import QAOA # This is the MinimumEigensolver factory
from qiskit_algorithms.optimizers import COBYLA, SPSA, NELDER_MEAD
from qiskit.primitives import Sampler

# Configure logging for this module
logging.getLogger('qiskit_algorithms.optimizers.spsa').setLevel(logging.WARNING)
logging.getLogger('qiskit_algorithms.minimum_eigensolvers.qaoa').setLevel(logging.WARNING) # Adjusted logger name
logger = logging.getLogger(__name__)

class QuantumOptimizer:
    """
    Optimizes the EnergyManagementProblem using QAOA to find a battery
    charge/discharge schedule that minimizes cost.
    """
    def __init__(self, problem, qaoa_reps, classical_optimizer_config):
        """
        Initializes the QuantumOptimizer.

        Args:
            problem (EnergyManagementProblem): The energy management problem instance.
            qaoa_reps (int): Number of QAOA repetitions (p-value).
            classical_optimizer_config (dict): Configuration for the classical optimizer
                                                 used within QAOA (e.g., {'name': 'COBYLA', 'parameters': {}}).
        """
        self.problem = problem
        self.num_slots = problem.num_slots
        self.penalty_factor = problem.penalty_factor
        self.qaoa_reps = qaoa_reps
        self.classical_optimizer_config = classical_optimizer_config
        self.qp = None

        logger.info(f"QuantumOptimizer initialized for {self.num_slots} slots with QAOA reps={self.qaoa_reps}, "
                    f"classical optimizer='{classical_optimizer_config['name']}' with parameters: {classical_optimizer_config.get('parameters', {})}.")

    def _build_qubo(self):
        """
        Builds the QuadraticProgram representation of the energy management problem.
        """
        logger.info("\n--- Building Quadratic Program for QAOA ---")
        
        self.qp = QuadraticProgram(name="energy_management_qubo")
        
        # --- Define Binary Variables ---
        # Variables are added directly to the QuadraticProgram instance.
        x_chg_vars = [self.qp.binary_var(name=f'x_chg_{t}') for t in range(self.num_slots)]
        x_dis_vars = [self.qp.binary_var(name=f'x_dis_{t}') for t in range(self.num_slots)]

        logger.info(f"Defined {self.qp.get_num_vars()} binary variables: "
                    f"{', '.join([var.name for var in self.qp.variables])}")

        # --- Objective Function and Constraints as Penalties ---
        objective_linear_coeffs = {}
        objective_quadratic_coeffs = {}
        objective_constant = 0.0

        # 1. Penalty for Simultaneous Charge/Discharge
        logger.info("\n--- Adding Objective: Simultaneous Charge/Discharge Penalties ---")
        for t in range(self.num_slots):
            key = (x_chg_vars[t].name, x_dis_vars[t].name) # Order doesn't strictly matter for QP obj dicts
            objective_quadratic_coeffs[key] = objective_quadratic_coeffs.get(key, 0.0) + self.penalty_factor
            logger.info(f"      Added {self.penalty_factor:.2f} * {x_chg_vars[t].name} * {x_dis_vars[t].name} "
                        f"for simultaneous C/D penalty at t={t}.")

        # 2. Grid Interaction Costs/Benefits
        logger.info("\n--- Adding Objective: Grid Interaction Costs/Benefits ---")
        eff = self.problem.battery_efficiency
        max_ex = self.problem.battery_max_exchange # Energy to/from battery terminals

        for t in range(self.num_slots):
            # Cost for charging: if x_chg_t=1, max_ex goes into battery.
            # Energy drawn from grid = max_ex / eff.
            cost_val_charge = (max_ex / eff) * self.problem.buy_price[t] if eff > 0 else float('inf')
            objective_linear_coeffs[x_chg_vars[t].name] = \
                objective_linear_coeffs.get(x_chg_vars[t].name, 0.0) + cost_val_charge
            logger.info(f"      Added {cost_val_charge:.4f} * {x_chg_vars[t].name} for charging cost at t={t}.")

            # Benefit from discharging: if x_dis_t=1, max_ex leaves battery.
            # Energy delivered to grid/load = max_ex * eff.
            benefit_val_discharge = -(max_ex * eff) * self.problem.sell_price[t]
            objective_linear_coeffs[x_dis_vars[t].name] = \
                objective_linear_coeffs.get(x_dis_vars[t].name, 0.0) + benefit_val_discharge
            logger.info(f"      Added {benefit_val_discharge:.4f} * {x_dis_vars[t].name} for discharging benefit at t={t}.")

        # 3. Battery State & Capacity Penalties
        logger.info("\n--- Adding Objective: Battery State & Capacity Penalties ---")
        # Assumed penalty form: P * [2*S_t_end^2 - 2*C_cap*S_t_end + C_cap^2] per time slot t
        # S_t_end = S_initial + sum_{k=0 to t} (x_chg_k * max_ex_stored_per_op - x_dis_k * max_ex_released_per_op)
        # Assuming max_ex is the amount transferred at battery terminal for state change calculation.
        
        coeff_chg_state = max_ex # Amount added to battery state for x_chg=1
        coeff_dis_state = -max_ex # Amount subtracted from battery state for x_dis=1
        C_cap = self.problem.battery_capacity
        P = self.penalty_factor

        current_S_const = self.problem.battery_initial_charge
        current_S_linear_coeffs = {} # Maps variable name to its coefficient in current S expression

        for t in range(self.num_slots):
            # Update S_t_end expression for current time slot t
            # S_t_end = current_S_const (from previous step or initial) + x_chg_t*coeff_chg_state + x_dis_t*coeff_dis_state
            S_t_end_const = current_S_const
            S_t_end_linear_coeffs = current_S_linear_coeffs.copy()
            
            S_t_end_linear_coeffs[x_chg_vars[t].name] = \
                S_t_end_linear_coeffs.get(x_chg_vars[t].name, 0.0) + coeff_chg_state
            S_t_end_linear_coeffs[x_dis_vars[t].name] = \
                S_t_end_linear_coeffs.get(x_dis_vars[t].name, 0.0) + coeff_dis_state
            
            # Penalty: P * [2*S_t_end^2 - 2*C_cap*S_t_end + C_cap^2]
            
            # Part 1: P * C_cap^2 (Constant)
            objective_constant += P * (C_cap**2)
            logger.info(f"      t={t}: Added constant {P * (C_cap**2):.4f} (from P*C_cap^2 term).")

            # Part 2: -2 * P * C_cap * S_t_end
            #   -2 * P * C_cap * S_t_end_const (Constant)
            objective_constant += -2 * P * C_cap * S_t_end_const
            logger.info(f"      t={t}: Added constant {-2 * P * C_cap * S_t_end_const:.4f} (from -2*P*C_cap*S_const term).")
            #   -2 * P * C_cap * Sum(coeff_i * Var_i) (Linear)
            for var_name, coeff in S_t_end_linear_coeffs.items():
                term_val = -2 * P * C_cap * coeff
                objective_linear_coeffs[var_name] = objective_linear_coeffs.get(var_name, 0.0) + term_val
                logger.info(f"      t={t}: Added linear {term_val:.4f}*{var_name} (from -2*P*C_cap*S_linear term).")
            
            # Part 3: 2 * P * S_t_end^2
            # S_t_end^2 = (S_t_end_const + Sum(L_i * Var_i))^2
            #           = S_t_end_const^2 + Sum(L_i^2 * Var_i) + 2*S_t_end_const*Sum(L_i*Var_i) + Sum_{i<j}(2*L_i*L_j*Var_i*Var_j)
            
            #   2 * P * S_t_end_const^2 (Constant)
            objective_constant += 2 * P * (S_t_end_const**2)
            logger.info(f"      t={t}: Added constant {2 * P * (S_t_end_const**2):.4f} (from 2*P*S_const^2 term).")
            
            #   Linear terms from 2*P*(Sum(L_i^2 * Var_i) + 2*S_t_end_const*Sum(L_i*Var_i))
            for var_name, L_i in S_t_end_linear_coeffs.items():
                term_val_sq = 2 * P * (L_i**2) # From L_i^2 * Var_i (since Var_i^2 = Var_i)
                term_val_cross_const = 2 * P * 2 * S_t_end_const * L_i # From 2*S_const*L_i*Var_i
                objective_linear_coeffs[var_name] = \
                    objective_linear_coeffs.get(var_name, 0.0) + term_val_sq + term_val_cross_const
                logger.info(f"      t={t}: Added linear {(term_val_sq + term_val_cross_const):.4f}*{var_name} (from 2*P*S_linear_sq_and_cross_const term).")

            #   Quadratic terms from 2*P*Sum_{i<j}(2*L_i*L_j*Var_i*Var_j)
            var_names_in_S_list = list(S_t_end_linear_coeffs.keys())
            for i in range(len(var_names_in_S_list)):
                for j in range(i + 1, len(var_names_in_S_list)):
                    var1_name = var_names_in_S_list[i]
                    var2_name = var_names_in_S_list[j]
                    L1 = S_t_end_linear_coeffs[var1_name]
                    L2 = S_t_end_linear_coeffs[var2_name]
                    
                    term_val_quad = 2 * P * 2 * L1 * L2
                    key = tuple(sorted((var1_name, var2_name)))
                    objective_quadratic_coeffs[key] = \
                        objective_quadratic_coeffs.get(key, 0.0) + term_val_quad
                    logger.info(f"      t={t}: Added quadratic {term_val_quad:.4f}*{var1_name}*{var2_name} (from 2*P*S_linear_cross_vars term).")
            
            # Prepare for next iteration: S_{t+1}_initial_state = S_t_end
            current_S_const = S_t_end_const
            current_S_linear_coeffs = S_t_end_linear_coeffs.copy()


        self.qp.minimize(constant=objective_constant,
                           linear=objective_linear_coeffs,
                           quadratic=objective_quadratic_coeffs)

        logger.info("\n--- Finalized QuadraticProgram Objective ---")
        logger.info(f"Objective Constant: {self.qp.objective.constant:.4f}")
        # Qiskit's .to_dict() for linear and quadratic parts gives {index: coeff}
        # To display with names:
        linear_display = {self.qp.get_variable(i).name: coeff for i, coeff in self.qp.objective.linear.to_dict().items()}
        quad_display_dict = {}
        for (i, j), coeff in self.qp.objective.quadratic.to_dict().items():
            quad_display_dict[(self.qp.get_variable(i).name, self.qp.get_variable(j).name)] = coeff
        logger.info(f"Linear Terms: {linear_display}")
        logger.info(f"Quadratic Terms: {quad_display_dict}")
        logger.info("Quadratic Program building complete.")


    def optimize(self):
        """
        Runs the QAOA optimization to find the optimal schedule.
        """
        self._build_qubo()
        if self.qp is None:
            logger.error("QuadraticProgram was not built. Aborting optimization.")
            return {'x_chg': [0]*self.num_slots, 'x_dis': [0]*self.num_slots, 'cost': float('inf'), 'status': "Error: QP not built"}

        logger.info("\n--- Running QAOA Algorithm ---")

        optimizer_name = self.classical_optimizer_config['name']
        optimizer_params = self.classical_optimizer_config.get('parameters', {})
        
        if optimizer_name.lower() == 'cobyla':
            classical_optimizer = COBYLA(**optimizer_params)
        elif optimizer_name.lower() == 'spsa':
            classical_optimizer = SPSA(**optimizer_params)
        elif optimizer_name.lower() == 'nelder_mead':
            classical_optimizer = NELDER_MEAD(**optimizer_params)
        else:
            raise ValueError(f"Unsupported classical optimizer: {optimizer_name}. Choose from COBYLA, SPSA, NELDER_MEAD.")
        
        qaoa_mes = QAOA(sampler=Sampler(), optimizer=classical_optimizer, reps=self.qaoa_reps)
        optimizer = MinimumEigenOptimizer(qaoa_mes) # Use the wrapper

        logger.info(f"QAOA configured with {self.qaoa_reps} repetition(s) and {optimizer_name} classical optimizer.")
        logger.info("Solving problem using MinimumEigenOptimizer with QAOA. This may take a moment...")
        
        opt_result = optimizer.solve(self.qp) # This returns an OptimizationResult

        logger.info("\n--- Quantum Optimization Result ---")
        
        status = opt_result.status.name # e.g., SUCCESS, FAILURE
        fval = opt_result.fval # QUBO objective value

        if opt_result.status.value == 0: # SUCCESS = 0
            # x is a list of variable values in order of qp.variables
            # variables_dict maps names to values
            x_chg_solution = [int(round(opt_result.variables_dict.get(f'x_chg_{t}', 0.0))) for t in range(self.num_slots)]
            x_dis_solution = [int(round(opt_result.variables_dict.get(f'x_dis_{t}', 0.0))) for t in range(self.num_slots)]
            log_status = "SUCCESS"
        else:
            logger.warning(f"QAOA optimization finished with status: {status}. Solution might not be optimal or found.")
            # Fallback or default solution
            x_chg_solution = [0] * self.num_slots
            x_dis_solution = [0] * self.num_slots
            log_status = status # Keep original status for logging

        # Log individual variable values (example for num_slots=2, generalize if needed)
        solution_vars_log = ", ".join([f"{var_name}={opt_result.variables_dict.get(var_name, 'N/A'):.1f}" 
                                   for var_name in [v.name for v in self.qp.variables]])
        logger.info(f"Objective function value (QUBO): {fval:.4f}")
        logger.info(f"Variable values: {solution_vars_log}")
        logger.info(f"Status: {log_status}")
        logger.info(f"QAOA Found Schedule: x_chg={x_chg_solution}, x_dis={x_dis_solution}")

        # Re-calculate cost using the EnergyManagementProblem's method
        schedule_dict = {'x_chg': x_chg_solution, 'x_dis': x_dis_solution}
        # Corrected line:
        actual_problem_cost = self.problem.calculate_full_cost_and_penalties(
            schedule_dict['x_chg'], 
            schedule_dict['x_dis']
            # verbose=False # Or True, or self.problem.verbose_trace, if you want trace from here too
        )

        logger.info(f"QAOA Found Schedule Cost (re-calculated using EnergyManagementProblem): ${actual_problem_cost:.4f}")
        logger.info("--- QUANTUM OPTIMIZATION COMPLETE ---")
        logger.info("=" * 70)

        return {'x_chg': x_chg_solution, 'x_dis': x_dis_solution, 'cost': actual_problem_cost, 'status': log_status}