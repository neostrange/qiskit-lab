import logging
import yaml
import argparse
import numpy as np
import os

# Suppress common NumPy runtime warnings (e.g., divide by zero)
np.seterr(all='ignore')

# Import core classes
from models.energy_problem import EnergyManagementProblem
from optimizers.classical_optimizer import ClassicalOptimizer
from optimizers.quantum_optimizer import QuantumOptimizer

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress verbose logs from Qiskit optimizers
logging.getLogger('qiskit_algorithms.optimizers.spsa').setLevel(logging.WARNING)
logging.getLogger('qiskit_algorithms.minimum_eigensolvers.sampling_vqe').setLevel(logging.WARNING)


def generate_random_problem_data(params, num_slots, seed=None):
    """
    Generate random load, solar, buy price, and sell price arrays based on
    normal distributions defined by base values and variances in params.

    Args:
        params (dict): Parameters for random generation (base values and variances).
        num_slots (int): Number of time slots.
        seed (int or None): Random seed for reproducibility.

    Returns:
        tuple: Four lists representing load, solar, buy_price, sell_price.
    """
    if seed is not None:
        np.random.seed(seed)

    load = np.maximum(
        0, np.random.normal(params.get('base_load', 5.0), params.get('load_variance', 1.0), num_slots)
    )
    solar = np.maximum(
        0, np.random.normal(params.get('base_solar', 4.0), params.get('solar_variance', 1.0), num_slots)
    )
    buy_price = np.maximum(
        0, np.random.normal(params.get('base_buy_price', 0.2), params.get('buy_price_variance', 0.05), num_slots)
    )
    sell_price = np.maximum(
        0, np.random.normal(params.get('base_sell_price', 0.1), params.get('sell_price_variance', 0.03), num_slots)
    )

    return load.tolist(), solar.tolist(), buy_price.tolist(), sell_price.tolist()


def run_classical_optimization(problem):
    logger.info("\n" + "=" * 70)
    logger.info("--- STARTING CLASSICAL OPTIMIZATION ---")
    classical_optimizer = ClassicalOptimizer(problem)
    result = classical_optimizer.optimize()
    logger.info("=" * 70)

    logger.info("\n" + "=" * 70)
    logger.info("--- VERIFYING CLASSICAL OPTIMAL SOLUTION COST ---")
    solution = {'x_chg': result['x_chg'], 'x_dis': result['x_dis']}
    logger.info(f"Re-calculating cost for classical solution: {solution}")
    verified_cost = problem.calculate_full_cost_and_penalties(
        solution['x_chg'], solution['x_dis'], verbose=True
    )

    if abs(result['cost'] - verified_cost) > EnergyManagementProblem.EPSILON:
        logger.warning(
            f"Classical optimizer's reported cost (${result['cost']:.2f}) "
            f"does NOT match verified cost (${verified_cost:.2f}). "
            f"Discrepancy: {abs(result['cost'] - verified_cost):.2f}. Investigate `ClassicalOptimizer`."
        )
    else:
        logger.info(f"Classical Optimal Solution VERIFIED Cost: ${verified_cost:.2f}")
    logger.info("=" * 70)
    return solution, verified_cost


def run_quantum_optimization(problem, qaoa_params, classical_optimizer_config):
    logger.info("\n" + "=" * 70)
    logger.info("--- STARTING QUANTUM OPTIMIZATION (QAOA on Simulator) ---")
    quantum_optimizer = QuantumOptimizer(
        problem=problem,
        qaoa_reps=qaoa_params.get('reps', 5),
        classical_optimizer_config=classical_optimizer_config
    )
    result = quantum_optimizer.optimize()
    logger.info("=" * 70)

    logger.info("\n" + "=" * 70)
    logger.info("--- VERIFYING QUANTUM OPTIMIZATION RESULT COST ---")
    solution = {'x_chg': result['x_chg'], 'x_dis': result['x_dis']}
    logger.info(f"Re-calculating cost for quantum solution: {solution}")
    verified_cost = problem.calculate_full_cost_and_penalties(
        solution['x_chg'], solution['x_dis'], verbose=True
    )

    if abs(result['cost'] - verified_cost) > EnergyManagementProblem.EPSILON:
        logger.warning(
            f"Quantum optimizer's reported cost (${result['cost']:.2f}) "
            f"does NOT match verified cost (${verified_cost:.2f}). "
            f"Discrepancy: {abs(result['cost'] - verified_cost):.2f}. "
            f"Check QUBO mapping or scaling."
        )
    else:
        logger.info(f"Quantum Found Solution VERIFIED Cost: ${verified_cost:.2f}")
    logger.info("=" * 70)
    return solution, verified_cost


def main():
    parser = argparse.ArgumentParser(description="Run Microgrid Energy Management Optimization.")
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to the configuration YAML file."
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found at: {args.config}")
        return

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded successfully from '{args.config}'.")
    except yaml.YAMLError as e:
        logger.error(f"Error loading YAML configuration: {e}")
        return
    except Exception as e:
        logger.error(f"Unexpected error loading config: {e}")
        return

    # Extract configuration sections
    random_data_config = config.get('random_data_generation', {})
    generate_random_data = random_data_config.get('generate_random_data', False)
    random_seed = random_data_config.get('random_seed', None)

    problem_params = config.get('problem_parameters', {})
    num_slots = problem_params.get('num_slots', 2)

    # Generate or load problem data
    if generate_random_data:
        load, solar, buy_price, sell_price = generate_random_problem_data(
            random_data_config, num_slots, random_seed
        )
        logger.info(f"Random problem data generated for {num_slots} slots.")
    else:
        load = problem_params.get('load', [5.0] * num_slots)
        solar = problem_params.get('solar', [4.0] * num_slots)
        buy_price = problem_params.get('buy_price', [0.20] * num_slots)
        sell_price = problem_params.get('sell_price', [0.10] * num_slots)

    # Battery and penalty parameters
    battery_capacity = problem_params.get('battery_capacity', 3.0)
    battery_initial_charge = problem_params.get('battery_initial_charge', 1.0)
    battery_max_exchange = problem_params.get('battery_max_exchange', 1.0)
    battery_efficiency = problem_params.get('battery_efficiency', 1.0)
    penalty_factor = problem_params.get('penalty_factor', 100000.0)

    # Initialize problem instance
    problem = EnergyManagementProblem(
        num_slots=num_slots,
        load=load,
        solar=solar,
        buy_price=buy_price,
        sell_price=sell_price,
        battery_capacity=battery_capacity,
        battery_initial_charge=battery_initial_charge,
        battery_max_exchange=battery_max_exchange,
        battery_efficiency=battery_efficiency,
        penalty_factor=penalty_factor,
        verbose_trace=False
    )

    logger.info(
        f"Problem Initialized: Penalty Factor={penalty_factor}, Num Slots={num_slots}, "
        f"Battery Capacity={battery_capacity} kWh, Initial Charge={battery_initial_charge} kWh"
    )
    logger.info(f"    Load: {load}")
    logger.info(f"    Solar: {solar}")
    logger.info(f"    Buy Price: {buy_price}")
    logger.info(f"    Sell Price: {sell_price}")

    # Run Classical Optimization
    classical_solution, classical_cost = run_classical_optimization(problem)

    # Run Quantum Optimization
    qaoa_params = config.get('qaoa_parameters', {})
    classical_optimizer_config = config.get('qaoa_classical_optimizer', {
        'name': 'SPSA',
        'parameters': {'maxiter': 1000}
    })

    quantum_solution, quantum_cost = run_quantum_optimization(
        problem, qaoa_params, classical_optimizer_config
    )

    # Final comparison
    logger.info("\n" + "=" * 70)
    logger.info("--- FINAL COMPARISON OF OPTIMIZER RESULTS ---")
    logger.info(f"Classical Best Solution: {classical_solution}, Cost: ${classical_cost:.2f}")
    logger.info(f"Quantum Found Solution: {quantum_solution}, Cost: ${quantum_cost:.2f}")

    difference = abs(classical_cost - quantum_cost)
    if difference > EnergyManagementProblem.EPSILON:
        logger.warning("WARNING: Quantum solution does NOT perfectly match classical optimal solution.")
        logger.warning("Possible reasons include approximate nature of QAOA, optimizer limitations,")
        logger.warning("or imperfect QUBO encoding of constraints.")
        logger.warning(f"Difference: ${difference:.2f}")
        logger.warning("Consider:")
        logger.warning(f"    1. Reviewing QUBO formulation in `QuantumOptimizer`.")
        logger.warning(f"    2. Increasing PENALTY_FACTOR (currently {penalty_factor}) in config.yaml.")
        logger.warning(f"    3. Increasing QAOA 'reps' (currently {qaoa_params.get('reps', 5)}) in config.yaml.")
        logger.warning(f"    4. Tuning QAOA classical optimizer parameters for '{classical_optimizer_config.get('name', 'SPSA')}'.")
    else:
        logger.info("SUCCESS: Quantum solution matches classical optimal solution.")

    logger.info("\n" + "=" * 70)
    logger.info("--- SIMULATION COMPLETE ---")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
