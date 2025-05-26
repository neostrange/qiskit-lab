"""
================================================================================
ProblemDataGenerator: Instance Creation for Battery Energy Management
================================================================================

This module provides the ProblemDataGenerator class, which generates problem
instances for simulation.

Domain Context:
---------------
- Supports both deterministic (from config) and randomized data generation.
- Allows reproducible experiments by setting random seeds.
- Facilitates benchmarking and robustness testing for optimizers.

================================================================================
"""


import numpy as np

class ProblemDataGenerator:
    """
    Generates problem parameters (load, solar, prices, battery specs)
    for the EnergyManagementProblem.
    """
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def generate_random_params(self, num_slots,
                               base_battery_capacity=3.0,
                               battery_capacity_variance=1.0,
                               base_initial_charge=1.0,
                               initial_charge_variance=0.5,
                               base_max_exchange=1.0,
                               max_exchange_variance=0.2,
                               penalty_factor=1000.0):
        """
        Generates random problem parameters for a given number of slots.
        """
        load = np.round(np.random.uniform(2, 10, num_slots), 1).tolist()
        solar = np.round(np.random.uniform(0, 8, num_slots), 1).tolist()
        buy_price = np.round(np.random.uniform(0.15, 0.40, num_slots), 2).tolist()
        sell_price = np.round(np.random.uniform(0.05, 0.20, num_slots), 2).tolist()

        battery_capacity = base_battery_capacity + np.random.uniform(-battery_capacity_variance / 2, battery_capacity_variance / 2)
        battery_initial_charge = base_initial_charge + np.random.uniform(-initial_charge_variance / 2, initial_charge_variance / 2)
        battery_max_exchange = base_max_exchange + np.random.uniform(-max_exchange_variance / 2, max_exchange_variance / 2)

        # Ensure non-negative and reasonable values for battery params
        battery_capacity = max(0.1, battery_capacity)
        battery_initial_charge = np.clip(battery_initial_charge, 0, battery_capacity)
        battery_max_exchange = max(0.1, battery_max_exchange)


        return {
            "num_slots": num_slots,
            "load": load,
            "solar": solar,
            "buy_price": buy_price,
            "sell_price": sell_price,
            "battery_capacity": battery_capacity,
            "battery_initial_charge": battery_initial_charge,
            "battery_max_exchange": battery_max_exchange,
            "battery_efficiency": 1.0, # Can make this variable too
            "penalty_factor": penalty_factor
        }

    # You could add other methods here, e.g.:
    # def generate_seasonal_data(self, num_slots, season_type): pass
    # def load_data_from_csv(self, filepath): pass