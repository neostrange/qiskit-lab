from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import yaml
import numpy as np
from models.energy_problem import EnergyManagementProblem
from optimizers.classical_optimizer import ClassicalOptimizer
from optimizers.quantum_optimizer import QuantumOptimizer
from main import generate_random_problem_data
import yaml
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

CONFIG_PATH = 'config.yaml'  # Adjust if needed




app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return "Quantum-Classical Optimization API is running."





@app.route('/run-optimization', methods=['POST'])
def run_optimization():
    try:
        config = request.get_json()
        if not config:
            return jsonify({"error": "No configuration data received."}), 400

        # Extract config
        random_data_config = config.get('random_data_generation', {})
        generate_random_data = random_data_config.get('generate_random_data', False)
        random_seed = random_data_config.get('random_seed', None)
        problem_params = config.get('problem_parameters', {})
        num_slots = problem_params.get('num_slots', 2)

        if generate_random_data:
            load, solar, buy_price, sell_price = generate_random_problem_data(
                random_data_config, num_slots, random_seed
            )
        else:
            load = problem_params.get('load', [5.0] * num_slots)
            solar = problem_params.get('solar', [4.0] * num_slots)
            buy_price = problem_params.get('buy_price', [0.20] * num_slots)
            sell_price = problem_params.get('sell_price', [0.10] * num_slots)

        # Init problem
        problem = EnergyManagementProblem(
            num_slots=num_slots,
            load=load,
            solar=solar,
            buy_price=buy_price,
            sell_price=sell_price,
            battery_capacity=problem_params.get('battery_capacity', 3.0),
            battery_initial_charge=problem_params.get('battery_initial_charge', 1.0),
            battery_max_exchange=problem_params.get('battery_max_exchange', 1.0),
            battery_efficiency=problem_params.get('battery_efficiency', 1.0),
            penalty_factor=problem_params.get('penalty_factor', 100000.0),
            verbose_trace=False
        )

        # Run classical
        classical = ClassicalOptimizer(problem).optimize()
        classical_cost = problem.calculate_full_cost_and_penalties(
            classical['x_chg'], classical['x_dis']
        )

        # Run quantum
        qaoa_params = config.get('qaoa_parameters', {})
        qaoa_optimizer_config = config.get('qaoa_classical_optimizer', {
            'name': 'SPSA',
            'parameters': {'maxiter': 1000}
        })
        quantum = QuantumOptimizer(problem, qaoa_params.get('reps', 5), qaoa_optimizer_config).optimize()
        quantum_cost = problem.calculate_full_cost_and_penalties(
            quantum['x_chg'], quantum['x_dis']
        )

        return jsonify({
            "input": {
                "load": load, "solar": solar,
                "buy_price": buy_price, "sell_price": sell_price
            },
            "classical": {
                "x_chg": classical['x_chg'],
                "x_dis": classical['x_dis'],
                "cost": round(classical_cost, 4)
            },
            "quantum": {
                "x_chg": quantum['x_chg'],
                "x_dis": quantum['x_dis'],
                "cost": round(quantum_cost, 4)
            },
            "difference": round(abs(classical_cost - quantum_cost), 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


@app.route('/api/random-problem', methods=['GET'])
def get_random_problem():
    try:
        # You can customize this if you want different randomness
        random_data_config = {
            "solar_mean": 3.0,
            "solar_std": 1.0,
            "load_mean": 4.0,
            "load_std": 1.5,
            "price_buy_range": [0.15, 0.30],
            "price_sell_range": [0.05, 0.15],
            "generate_random_data": True
        }
        num_slots = 4
        seed = None  # or set a fixed integer for reproducibility

        load, solar, buy_price, sell_price = generate_random_problem_data(
            random_data_config, num_slots, seed
        )

        return jsonify({
            "load": load,
            "solar": solar,
            "buy_price": buy_price,
            "sell_price": sell_price
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f) or {}

def save_config(config_data):
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config_data, f)

def deep_update(original, updates):
    """
    Recursively update a dict with another dict (used for partial config updates).
    """
    for k, v in updates.items():
        if isinstance(v, dict) and k in original:
            deep_update(original[k], v)
        else:
            original[k] = v
    return original

@app.route('/api/config', methods=['GET'])
def get_config():
    try:
        config_data = load_config()
        return jsonify(config_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/update-config', methods=['POST'])
def update_config():
    try:
        new_config = request.get_json()
        if not new_config:
            return jsonify({"error": "No data received"}), 400

        current_config = load_config()
        updated_config = deep_update(current_config, new_config)
        save_config(updated_config)

        return jsonify({"message": "Config updated successfully", "config": updated_config})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/problem-parameters', methods=['GET'])
def get_problem_parameters():
    try:
        config = load_config()
        problem_params = config.get('problem_parameters', {})
        return jsonify(problem_params)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/manual-problem', methods=['POST'])
def set_manual_problem():
    try:
        manual_params = request.get_json()
        if not manual_params:
            return jsonify({"error": "No problem data received"}), 400

        config = load_config()
        config['problem_parameters'] = manual_params
        save_config(config)

        return jsonify({"message": "Manual problem parameters updated", "problem_parameters": manual_params})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    try:
        # You can customize this as needed
        return jsonify({
            "status": "running",
            "message": "Quantum-Classical Optimization backend is operational"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_config():
    try:
        # You can choose to reset to a default config file or an empty dict
        default_config = {
            "random_data_generation": {
                "generate_random_data": True,
                "random_seed": None
            },
            "problem_parameters": {
                "num_slots": 4,
                "load": [5.0, 5.0, 5.0, 5.0],
                "solar": [4.0, 4.0, 4.0, 4.0],
                "buy_price": [0.20, 0.20, 0.20, 0.20],
                "sell_price": [0.10, 0.10, 0.10, 0.10],
                "battery_capacity": 3.0,
                "battery_initial_charge": 1.0,
                "battery_max_exchange": 1.0,
                "battery_efficiency": 1.0,
                "penalty_factor": 100000.0
            },
            "qaoa_parameters": {
                "reps": 5
            },
            "qaoa_classical_optimizer": {
                "name": "SPSA",
                "parameters": {
                    "maxiter": 1000
                }
            }
        }
        save_config(default_config)
        return jsonify({"message": "Config reset to default", "config": default_config})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/run-optimization-saved', methods=['GET'])
def run_optimization_saved():
    try:
        config = load_config()

        # Extract parts like your existing /run-optimization endpoint
        random_data_config = config.get('random_data_generation', {})
        generate_random_data = random_data_config.get('generate_random_data', False)
        random_seed = random_data_config.get('random_seed', None)
        problem_params = config.get('problem_parameters', {})
        num_slots = problem_params.get('num_slots', 2)

        if generate_random_data:
            load, solar, buy_price, sell_price = generate_random_problem_data(
                random_data_config, num_slots, random_seed
            )
        else:
            load = problem_params.get('load', [5.0] * num_slots)
            solar = problem_params.get('solar', [4.0] * num_slots)
            buy_price = problem_params.get('buy_price', [0.20] * num_slots)
            sell_price = problem_params.get('sell_price', [0.10] * num_slots)

        problem = EnergyManagementProblem(
            num_slots=num_slots,
            load=load,
            solar=solar,
            buy_price=buy_price,
            sell_price=sell_price,
            battery_capacity=problem_params.get('battery_capacity', 3.0),
            battery_initial_charge=problem_params.get('battery_initial_charge', 1.0),
            battery_max_exchange=problem_params.get('battery_max_exchange', 1.0),
            battery_efficiency=problem_params.get('battery_efficiency', 1.0),
            penalty_factor=problem_params.get('penalty_factor', 100000.0),
            verbose_trace=False
        )

        classical = ClassicalOptimizer(problem).optimize()
        classical_cost = problem.calculate_full_cost_and_penalties(
            classical['x_chg'], classical['x_dis']
        )

        qaoa_params = config.get('qaoa_parameters', {})
        qaoa_optimizer_config = config.get('qaoa_classical_optimizer', {
            'name': 'SPSA',
            'parameters': {'maxiter': 1000}
        })
        quantum = QuantumOptimizer(problem, qaoa_params.get('reps', 5), qaoa_optimizer_config).optimize()
        quantum_cost = problem.calculate_full_cost_and_penalties(
            quantum['x_chg'], quantum['x_dis']
        )

        return jsonify({
            "input": {
                "load": load, "solar": solar,
                "buy_price": buy_price, "sell_price": sell_price
            },
            "classical": {
                "x_chg": classical['x_chg'],
                "x_dis": classical['x_dis'],
                "cost": round(classical_cost, 4)
            },
            "quantum": {
                "x_chg": quantum['x_chg'],
                "x_dis": quantum['x_dis'],
                "cost": round(quantum_cost, 4)
            },
            "difference": round(abs(classical_cost - quantum_cost), 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
