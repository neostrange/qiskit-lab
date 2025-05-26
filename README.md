# Quantum-Classical Energy Optimization Demo

This project demonstrates and compares quantum-inspired and classical optimization approaches for efficient energy management in a microgrid setting. It provides an interactive web interface to configure problem parameters, define energy scenarios (load, solar, prices), run optimization algorithms, and visualize the results.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Features](#features)
* [Project Structure](#project-structure)
* [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
    * [Running the Application](#running-the-application)
* [Usage](#usage)
* [Optimization Approaches](#optimization-approaches)
    * [Classical Optimization](#classical-optimization)
    * [Quantum Optimization (QAOA)](#quantum-optimization-qaoa)
* [API Endpoints](#api-endpoints)
* [Contributing](#contributing)
* [License](#license)

---

## Project Overview

Modern energy systems, particularly **microgrids** (e.g., homes with solar panels and batteries), face the complex challenge of efficiently managing energy resources. This involves balancing energy generation, demand, storage, and interactions with the main power grid, often under fluctuating electricity prices.

This interactive demonstration aims to find the **optimal schedule for charging and discharging a battery to minimize overall energy costs**. We explore and compare two primary approaches:

1.  **Classical Optimization:** Utilizing traditional algorithms to find optimal solutions for small-scale problems, serving as a benchmark.
2.  **Quantum-Inspired & Quantum Optimization:** Formulating the problem as a **Quadratic Unconstrained Binary Optimization (QUBO)** problem and employing the **Quantum Approximate Optimization Algorithm (QAOA)**, run on a classical simulator. This showcases the methodology for potential future application on more powerful quantum devices.

---

## Features

* **Interactive Web UI:** Configure and manage problem parameters via a user-friendly interface.
* **Dynamic Problem Definition:** Use server-side configurations, generate random problem data, or manually input specific energy scenarios.
* **Classical Optimizer:** Find exact optimal solutions for small problem instances using exhaustive search.
* **Quantum Optimizer (QAOA):** Apply QAOA on a simulated quantum environment to solve the energy optimization problem formulated as a QUBO.
* **Real-time Results:** View and compare the schedules and costs from both classical and quantum optimization runs.
* **Configuration Management:** Fetch, update, reset, upload, and download backend configuration files directly from the UI.
* **API Utilities:** Explore server status, configuration keys, and validate problem parameters through dedicated utility functions.

---

## Project Structure

The project is structured to separate concerns, making it modular and easier to understand:


├── analysis/
│   └── solution_analyzer.py      # Logic for analyzing and comparing optimization solutions
├── app.py                        # Main Flask application file defining API routes
├── config.yaml                   # Default configuration for problem parameters and QAOA
├── main.py                       # Entry point for running the Flask application
├── models/
│   └── energy_problem.py         # Defines the data structures for the energy optimization problem
├── notebooks/
│   └── ...                       # Jupyter notebooks for experimentation and analysis (e.g., Qiskit examples)
├── optimizers/
│   ├── classical_optimizer.py    # Implements classical optimization algorithms (e.g., exhaustive search)
│   └── quantum_optimizer.py      # Implements quantum optimization algorithms (e.g., QAOA with Qiskit)
├── qiskit-lab/
│   └── ...                       # Additional Qiskit-related lab files and examples
├── static/
│   └── style.css                 # Custom CSS for the frontend
├── templates/
│   └── index.html                # The main frontend HTML file (user interface)
└── utils/
└── problem_data_generator.py # Utility functions for generating random problem data


* **`app.py`**: The core Flask application defining the API endpoints.
* **`config.yaml`**: Stores default and user-modifiable parameters for the energy problem and QAOA.
* **`main.py`**: The script to run the Flask development server.
* **`models/energy_problem.py`**: Likely contains classes or functions to represent the energy system, such as `Battery`, `Load`, `Solar`, and their interactions.
* **`optimizers/classical_optimizer.py`**: Implements classical optimization algorithms (e.g., brute force) to find exact optimal solutions for small problems.
* **`optimizers/quantum_optimizer.py`**: Houses the quantum optimization logic, specifically the QAOA implementation using Qiskit to solve the QUBO formulation of the energy problem.
* **`analysis/solution_analyzer.py`**: Responsible for processing and comparing the outputs of both optimizers.
* **`utils/problem_data_generator.py`**: Utility functions for creating synthetic load, solar, and price data.
* **`templates/index.html`**: The single-page application (SPA) serving as the user interface, built with Bootstrap and JavaScript.
* **`static/style.css`**: Custom CSS rules to enhance the appearance.
* **`notebooks/` and `qiskit-lab/`**: These directories likely contain development notebooks and scripts used during the research and development phases, offering insights into the quantum computing aspects and problem formulation.

---

## Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

* **Python 3.8+**
* **`pip`** (Python package installer)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/your-username/qiskit_demo.git](https://github.com/your-username/qiskit_demo.git) # Replace with your actual repo URL
    cd qiskit_demo
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    # On Windows:
    # .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required Python packages:**

    ```bash
    pip install -r requirements.txt
    ```
    *If `requirements.txt` is missing, you can create it by running `pip freeze > requirements.txt` after manually installing dependencies like `Flask`, `Qiskit`, `PyYAML`, `numpy`, etc.*

### Running the Application

1.  **Start the Flask backend server:**

    ```bash
    python main.py
    ```
    The server should start on `http://localhost:5000`.

2.  **Open the frontend in your browser:**
    Navigate to `http://localhost:5000/` in your web browser.

---

## Usage

The web interface is designed to guide you through the demonstration:

1.  **1. Overview Tab:** Understand the project's purpose, the energy management problem, and the optimization approaches.
2.  **2. Configuration Tab:**
    * View and modify global parameters (e.g., battery capacity, `num_slots`, QAOA repetitions).
    * **Fetch**, **Update**, **Reset**, **Download**, or **Upload** the `config.yaml` file.
    * Click "Update Config on Server" to save changes.
3.  **3. Define Problem Tab:**
    * Choose the source for your problem data (load, solar, buy/sell prices) for the next optimization run:
        * **Option 1: Use Server Configuration:** Uses data from `config.yaml`.
        * **Option 2: Generate New Random Data:** Create a new random problem instance.
        * **Option 3: Define Manual Data:** Input custom comma-separated values.
    * The "Currently Active Problem Data" shows the data for the next run.
4.  **4. Run Optimization Tab:**
    * Click **"Run with Active Problem & Server Config"** to execute both optimizers.
    * **"Run with Active Problem & UI Config (Advanced)"** sends current UI form values for a one-time run.
    * Results, including schedules and costs, will be displayed. You can also fetch or clear the last stored result.
5.  **5. API Utilities Tab:**
    * Access diagnostic tools: Check **server status**, view **config keys**, and **validate** custom parameters.

---

## Optimization Approaches

### Classical Optimization

For small problem instances (e.g., 2-4 time slots), an **exhaustive search** algorithm is used. This method systematically checks every possible combination of battery charge/discharge decisions to guarantee finding the absolute minimum cost solution. While guaranteed optimal, this approach does not scale to larger numbers of time slots due to its exponential complexity.

### Quantum Optimization (QAOA)

The energy optimization problem is formulated as a **Quadratic Unconstrained Binary Optimization (QUBO)** problem. This involves transforming the objective (minimizing cost) and constraints (battery limits, charge balance) into a quadratic polynomial of binary variables.

The **Quantum Approximate Optimization Algorithm (QAOA)** is then employed:
* It's a hybrid quantum-classical algorithm.
* A quantum computer (or simulator) prepares and measures quantum states corresponding to the QUBO problem.
* A classical optimizer iteratively adjusts parameters of the quantum circuit to find the approximate ground state, which corresponds to the lowest-cost solution.

**Note:** For the problem sizes handled in this demo, the quantum optimization is performed on a classical simulator (e.g., using Qiskit's Aer simulator). While this demonstrates the methodology, a true quantum advantage is not expected on current noisy quantum hardware for these small problems. The potential lies in scaling to much larger and more complex energy management scenarios on future, more powerful quantum devices.

---

## API Endpoints

The Flask backend exposes several RESTful API endpoints:

* `GET /api/status`: Check the server's operational status.
* `GET /api/config`: Retrieve the current backend configuration (`config.yaml`).
* `POST /api/config`: Update the backend configuration with new settings.
* `POST /api/config/reset`: Reset the backend configuration to its default values.
* `GET /api/config/download`: Download the `config.yaml` file.
* `POST /api/config/upload`: Upload a new `config.yaml` file.
* `GET /api/config/keys`: Get the top-level keys available in the configuration.
* `POST /api/problem/generate_random`: Generate and return new random problem data based on specified `num_slots`.
* `POST /api/problem/validate`: Validate a set of problem parameters (load, solar, prices).
* `POST /api/optimize/run_with_server_config`: Run optimization using active problem data and server-side configuration.
* `POST /api/optimize/run_with_custom_config`: Run optimization with problem and configuration data provided in the request body.
* `GET /api/results/last`: Fetch the last stored optimization result.
* `POST /api/results/clear`: Clear the last stored optimization result.

---

## Contributing

Contributions are welcome! Please feel free to:

* Fork the repository.
* Create a new branch (`git checkout -b feature/your-feature-name`).
* Make your changes and commit them (`git commit -m 'Add new feature'`).
* Push to the branch (`git push origin feature/your-feature-name`).
* Open a Pull Request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
