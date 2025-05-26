"""
================================================================================
EnergyManagementProblem: Mathematical Model for Microgrid Battery Optimization
================================================================================

This module defines the EnergyManagementProblem class, which encapsulates the
formulation of the battery energy management problem for a microgrid. It serves
as the central definition of the optimization challenge, providing parameters
and a robust method for calculating the total cost (including penalties) for
any given battery charge/discharge schedule.

Domain Context:
---------------
The problem models a microgrid system equipped with:
- **Time-varying Electricity Load (Demand):** How much power the household/building needs at specific times.
- **Solar Generation:** Power produced by local solar panels, which varies with time (e.g., day/night).
- **Energy Storage Battery:** A battery with defined capacity, initial charge, and maximum charge/discharge rate.
- **Grid Connection:** Ability to buy electricity from the main grid at varying prices and sell excess electricity back to the grid at different prices.

The core objective is to determine an optimal battery charge/discharge schedule
across multiple time slots (e.g., hours in a day) that minimizes the total
electricity cost. This cost includes money spent buying from the grid, money
earned from selling to the grid, and severe penalties for violating operational
constraints of the battery.

Key Constraints Modeled:
------------------------
1.  **Battery Capacity Limits:** The battery's charge level must always remain
    between 0 (empty) and its maximum capacity.
2.  **Charge/Discharge Exclusivity:** At any given time slot, the battery
    cannot simultaneously charge and discharge.
3.  **Maximum Exchange Rate:** The amount of energy charged or discharged
    in a single time slot cannot exceed a predefined limit.
4.  **Battery Efficiency:** Energy conversion losses are accounted for during
    both charging and discharging.

This class is foundational for both the `ClassicalOptimizer` (for exact brute-force
verification) and the `QuantumOptimizer` (which translates this problem into a QUBO
for quantum solution).

================================================================================
"""

import numpy as np
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)

class EnergyManagementProblem:
    """
    Encapsulates all parameters defining the battery energy management problem.

    Attributes:
        num_slots (int): Number of time slots for optimization (e.g., hours in a day).
        load (np.array): Electrical load demand for each time slot (kWh).
        solar (np.array): Solar generation for each time slot (kWh).
        buy_price (np.array): Price to buy electricity from the grid for each slot ($/kWh).
        sell_price (np.array): Price to sell electricity to the grid for each slot ($/kWh).
        battery_capacity (float): Maximum energy the battery can store (kWh).
        battery_initial_charge (float): Initial charge level of the battery at t=0 (kWh).
        battery_max_exchange (float): Maximum energy that can be charged/discharged in one slot (kWh).
        battery_efficiency (float): Efficiency of battery energy conversion (0.0 to 1.0).
                                    Energy IN = Exchange * Efficiency (for charging)
                                    Energy OUT = Exchange / Efficiency (for discharging)
        penalty_factor (float): Multiplier for constraint violation penalties. A high value
                                encourages the optimizer to avoid violations.
        verbose_trace (bool): If True, enables detailed logging during cost calculation.
    """

    # Using a small epsilon for floating-point comparisons to avoid issues with near-zero values
    EPSILON = 1e-9

    def __init__(self, num_slots, load, solar, buy_price, sell_price,
                 battery_capacity, battery_initial_charge, battery_max_exchange,
                 battery_efficiency, penalty_factor,
                 verbose_trace=False):
        """
        Initializes the EnergyManagementProblem with given parameters.

        Args:
            num_slots (int): Number of time slots.
            load (list or np.array): Load values for each slot.
            solar (list or np.array): Solar generation values for each slot.
            buy_price (list or np.array): Buy prices for each slot.
            sell_price (list or np.array): Sell prices for each slot.
            battery_capacity (float): Battery's maximum capacity.
            battery_initial_charge (float): Battery's initial charge.
            battery_max_exchange (float): Max energy exchange per slot.
            battery_efficiency (float): Battery efficiency (0.0 to 1.0).
            penalty_factor (float): Multiplier for penalties.
            verbose_trace (bool): If True, enables detailed logging during cost calculation.

        Raises:
            ValueError: If input parameters are invalid (e.g., inconsistent lengths,
                        efficiency out of range, non-positive penalty factor).
        """
        self.num_slots = num_slots
        self.load = np.array(load, dtype=float)
        self.solar = np.array(solar, dtype=float)
        self.buy_price = np.array(buy_price, dtype=float)
        self.sell_price = np.array(sell_price, dtype=float)
        self.battery_capacity = float(battery_capacity)
        self.battery_initial_charge = float(battery_initial_charge)
        self.battery_max_exchange = float(battery_max_exchange)

        if not (0.0 <= battery_efficiency <= 1.0):
            logger.error(f"Invalid battery_efficiency: {battery_efficiency}. Must be between 0.0 and 1.0.")
            raise ValueError("Battery efficiency must be between 0.0 and 1.0.")
        self.battery_efficiency = float(battery_efficiency)

        if penalty_factor <= 0:
            logger.error(f"Invalid penalty_factor: {penalty_factor}. Must be a positive value.")
            raise ValueError("Penalty factor must be a positive value.")
        self.penalty_factor = float(penalty_factor)
        
        self.verbose_trace = verbose_trace

        if not (len(self.load) == len(self.solar) ==
                len(self.buy_price) == len(self.sell_price) == num_slots):
            logger.error("Mismatched time-series data lengths.")
            raise ValueError("All time-series data (load, solar, buy_price, sell_price) "
                             "must have length equal to num_slots.")

        logger.debug(f"EnergyManagementProblem initialized for {num_slots} slots.")
        logger.debug(f"Battery parameters: Capacity={self.battery_capacity} kWh, "
                      f"Initial Charge={self.battery_initial_charge} kWh, "
                      f"Max Exchange={self.battery_max_exchange} kWh/slot, "
                      f"Efficiency={self.battery_efficiency}.")
        logger.debug(f"Penalty Factor: {self.penalty_factor}")
        logger.debug(f"Verbose Trace enabled: {self.verbose_trace}")


    def get_params(self):
        """
        Returns a dictionary of all problem parameters.
        """
        return {
            "num_slots": self.num_slots,
            "load": self.load.tolist(),
            "solar": self.solar.tolist(),
            "buy_price": self.buy_price.tolist(),
            "sell_price": self.sell_price.tolist(),
            "battery_capacity": self.battery_capacity,
            "battery_initial_charge": self.battery_initial_charge,
            "battery_max_exchange": self.battery_max_exchange,
            "battery_efficiency": self.battery_efficiency,
            "penalty_factor": self.penalty_factor,
            "verbose_trace": self.verbose_trace
        }

    def calculate_full_cost_and_penalties(self, x_chg_vals, x_dis_vals, verbose=None):
        """
        Calculates the total monetary cost for a given battery charge/discharge schedule.
        This calculation includes both direct grid interaction costs/benefits and
        penalties for violating operational constraints (e.g., overcharge, undercharge,
        simultaneous charge/discharge).

        This method serves as the ground truth (classical cost function) against
        which the results from quantum optimizers are compared and validated.

        Args:
            x_chg_vals (list): List of binary (0/1) charge decisions for each time slot.
                               x_chg_vals[t] = 1 means charge battery at slot t.
            x_dis_vals (list): List of binary (0/1) discharge decisions for each time slot.
                               x_dis_vals[t] = 1 means discharge battery at slot t.
            verbose (bool, optional): If True, prints a detailed step-by-step execution trace
                                      of the calculations for each time slot. If None, uses
                                      the instance's `self.verbose_trace` setting.

        Returns:
            float: The total calculated cost, which is the sum of grid costs/benefits
                   and accumulated penalties.
        """
        total_grid_cost = 0.0
        total_penalties = 0.0
        current_battery_state = self.battery_initial_charge

        current_verbose_setting = verbose if verbose is not None else self.verbose_trace

        if current_verbose_setting:
            logger.info(f"\n" + "="*70)
            logger.info(f"--- TRACING SOLUTION --- Penalty Factor: {self.penalty_factor:.2f} ---")
            logger.info(f"Initial Battery Charge (t=0 start): {self.battery_initial_charge:.2f} kWh")
            logger.info("-" * 70)

        for t in range(self.num_slots):
            x_chg_t = x_chg_vals[t]
            x_dis_t = x_dis_vals[t]

            # Ensure inputs are treated as binary for calculation consistency
            x_chg_t = round(x_chg_t)
            x_dis_t = round(x_dis_t)

            # --- 1. Simultaneous Charge/Discharge Penalty ---
            if x_chg_t == 1 and x_dis_t == 1:
                penalty = self.penalty_factor
                total_penalties += penalty
                if current_verbose_setting: logger.info(f"Slot {t}: !!! PENALTY: Simultaneous C/D detected (x_chg_t=1, x_dis_t=1). Added: ${penalty:.2f}")

            # Calculate actual energy transferred to/from the battery in this slot.
            # Efficiency is applied here:
            energy_charged_to_battery = x_chg_t * self.battery_max_exchange * self.battery_efficiency
            energy_discharged_from_battery = x_dis_t * self.battery_max_exchange / self.battery_efficiency

            # --- Battery State Update ---
            next_battery_state = current_battery_state + energy_charged_to_battery - energy_discharged_from_battery

            # --- Battery Capacity Constraint Penalties ---
            # Over-capacity penalty
            overcharge_violation = max(0, next_battery_state - self.battery_capacity)
            if overcharge_violation > self.EPSILON:
                penalty = self.penalty_factor * (overcharge_violation ** 2)
                total_penalties += penalty
                if current_verbose_setting: logger.info(f"Slot {t}: !!! PENALTY: Overcharge ({next_battery_state:.2f} kWh > {self.battery_capacity:.2f} kWh). Added: ${penalty:.2f}")

            # Under-capacity (below zero) penalty
            undercharge_violation = max(0, -next_battery_state) # How much it went below zero
            if undercharge_violation > self.EPSILON:
                penalty = self.penalty_factor * (undercharge_violation ** 2)
                total_penalties += penalty
                if current_verbose_setting: logger.info(f"Slot {t}: !!! PENALTY: Undercharge ({next_battery_state:.2f} kWh < 0 kWh). Added: ${penalty:.2f}")

            # --- Calculate Net Energy Interaction with the Grid ---
            # This is the corrected calculation for grid interaction based on actual energy flows
            # after considering battery efficiency.
            # Positive value means energy needed from grid (buying)
            # Negative value means energy available to sell to grid (surplus)
            grid_interaction_kwh = (self.load[t] - self.solar[t]) # Local net demand/surplus
            grid_interaction_kwh += energy_charged_to_battery    # If charging, we draw more from grid
            grid_interaction_kwh -= energy_discharged_from_battery # If discharging, we supply to load/grid

            # --- Grid Interaction Costs/Benefits ---
            slot_cost = 0.0
            if grid_interaction_kwh > self.EPSILON: # Positive value means energy needed from grid (buying)
                slot_cost = grid_interaction_kwh * self.buy_price[t]
                if current_verbose_setting: logger.info(f"Slot {t}: Buying {grid_interaction_kwh:.2f} kWh from grid at ${self.buy_price[t]:.2f}/kWh. Cost: ${slot_cost:.2f}")
            elif grid_interaction_kwh < -self.EPSILON: # Negative value means energy available to sell to grid
                # grid_interaction_kwh is negative, so multiplying by sell_price gives a negative cost (a benefit)
                slot_cost = grid_interaction_kwh * self.sell_price[t] # This is a negative cost (revenue)
                if current_verbose_setting: logger.info(f"Slot {t}: Selling {-grid_interaction_kwh:.2f} kWh to grid at ${self.sell_price[t]:.2f}/kWh. Benefit: ${-slot_cost:.2f}")
            else: # Net zero interaction with epsilon tolerance
                if current_verbose_setting: logger.info(f"Slot {t}: No net grid interaction (within epsilon).")

            total_grid_cost += slot_cost
            current_battery_state = next_battery_state

            # --- Verbose Trace Output for the Current Slot ---
            if current_verbose_setting:
                logger.info(f"Slot {t}: Decisions: Chg={int(x_chg_t)}, Dis={int(x_dis_t)}")
                logger.info(f"  Local Net (Load-Solar): {(self.load[t] - self.solar[t]):.2f} kWh")
                logger.info(f"  Battery: Effective Charge {energy_charged_to_battery:.2f} kWh, Effective Discharge {energy_discharged_from_battery:.2f} kWh")
                logger.info(f"  Battery State: {current_battery_state - (energy_charged_to_battery - energy_discharged_from_battery):.2f} kWh (Start) -> {current_battery_state:.2f} kWh (End) (Capacity: {self.battery_capacity:.2f} kWh)")
                logger.info(f"  Net Grid Interaction (after battery): {grid_interaction_kwh:.2f} kWh")
                logger.info(f"  Slot Grid Cost/Benefit: ${slot_cost:.2f}")
                logger.info(f"  Cumulative Penalties: ${total_penalties:.2f}")
                logger.info(f"  Current Total Cost (Grid + Penalties): ${total_grid_cost + total_penalties:.2f}")
                logger.info("-" * 70)

        if current_verbose_setting:
            logger.info(f"--- END OF SOLUTION TRACE ---")
            logger.info(f"Summary:")
            logger.info(f"  Total Grid Cost (from buying/selling): ${total_grid_cost:.2f}")
            logger.info(f"  Total Accumulated Penalties: ${total_penalties:.2f}")
            logger.info(f"  Overall Objective Value (Total Cost): ${total_grid_cost + total_penalties:.2f}")
            logger.info("="*70 + "\n")

        return total_grid_cost + total_penalties