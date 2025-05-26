"""
================================================================================
ClassicalOptimizer: Brute-Force Solution for Microgrid Battery Optimization
================================================================================

This module defines the ClassicalOptimizer class, which solves the
EnergyManagementProblem using a brute-force approach. It systematically explores
every possible combination of binary charge/discharge decisions across all time
slots to find the schedule that yields the absolute minimum cost.

While computationally intensive and impractical for large numbers of time slots
(due to exponential complexity), this classical brute-force method serves as the
"ground truth" or benchmark. It guarantees finding the global optimum for the
given problem parameters and is essential for validating the approximate
solutions found by quantum optimizers like QAOA.

The class interacts directly with the `EnergyManagementProblem` to evaluate the
cost of each potential schedule using its `calculate_full_cost_and_penalties`
method.

================================================================================
"""

import numpy as np
import itertools
import logging

logger = logging.getLogger(__name__)

class ClassicalOptimizer:
    """
    Solves the EnergyManagementProblem using a brute-force approach to find
    the optimal battery charge/discharge schedule.
    """
    def __init__(self, problem):
        """
        Initializes the ClassicalOptimizer with a given energy management problem.

        Args:
            problem (EnergyManagementProblem): The energy management problem instance.
        """
        self.problem = problem
        self.num_slots = problem.num_slots
        logger.info(f"ClassicalOptimizer initialized for {self.num_slots} slots.")

    def optimize(self):
        """
        Performs a brute-force search over all possible charge/discharge schedules
        to find the one with the minimum total cost.

        Returns:
            dict: A dictionary containing the optimal charge and discharge schedules
                  and the corresponding minimum cost.
                  Example: {'x_chg': [0, 1], 'x_dis': [1, 0], 'cost': 123.45}
        """
        logger.info(f"\n--- Starting Brute-Force Classical Optimization for {self.num_slots} slots ---")
        
        min_cost = float('inf')
        optimal_chg_schedule = None
        optimal_dis_schedule = None
        total_combinations = 2**(2 * self.num_slots) # 2 choices (0/1) for x_chg, 2 for x_dis, per slot.
                                                     # So 2 * num_slots variables, each can be 0 or 1.

        logger.info(f"Evaluating {total_combinations} possible combinations...")

        # Generate all binary combinations for x_chg and x_dis schedules
        # For N slots, we have N x_chg variables and N x_dis variables.
        # So, 2N binary variables in total.
        # itertools.product generates all combinations of (0,1) for 2N positions.
        for combination in itertools.product([0, 1], repeat=2 * self.num_slots):
            # Split the combination into x_chg_vals and x_dis_vals
            x_chg_vals = list(combination[:self.num_slots])
            x_dis_vals = list(combination[self.num_slots:])

            # Calculate the cost for the current schedule
            current_cost = self.problem.calculate_full_cost_and_penalties(
                x_chg_vals=x_chg_vals,
                x_dis_vals=x_dis_vals,
                verbose=False # Keep verbose off during brute-force to avoid excessive logging
            )

            # Update if a lower cost is found
            if current_cost < min_cost:
                min_cost = current_cost
                optimal_chg_schedule = x_chg_vals
                optimal_dis_schedule = x_dis_vals
                logger.debug(f"New minimum found: x_chg={optimal_chg_schedule}, x_dis={optimal_dis_schedule}, Cost=${min_cost:.2f}")

        if optimal_chg_schedule is None:
            logger.error("No valid solution found. This should not happen for a brute-force search unless problem constraints are impossible.")
            return {'x_chg': [], 'x_dis': [], 'cost': float('inf'), 'status': 'FAILED'}

        logger.info(f"\n--- Classical Optimization Complete ---")
        logger.info(f"Optimal Classical Schedule: x_chg={optimal_chg_schedule}, x_dis={optimal_dis_schedule}")
        logger.info(f"Minimum Classical Cost Found: ${min_cost:.2f}")
        logger.info("---------------------------------------")

        return {
            'x_chg': optimal_chg_schedule,
            'x_dis': optimal_dis_schedule,
            'cost': min_cost,
            'status': 'SUCCESS'
        }