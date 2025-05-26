from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram
import numpy as np

# Define the item we are searching for (index)
target_item = 2  # Let's say we are looking for the item at index 2 (binary '10')

# Number of qubits needed to represent the search space
n_qubits = 2

# Create a quantum circuit
qc = QuantumCircuit(n_qubits, n_qubits)

# Initialize the state to a uniform superposition
qc.h(range(n_qubits))

# The Oracle (marks the target item)
oracle = QuantumCircuit(n_qubits, name="Oracle")
if target_item == 0:
    oracle.cz(0, 1)
elif target_item == 1:
    oracle.cz(0, 1)
    oracle.x(0)
elif target_item == 2:
    oracle.x(1)
    oracle.cz(0, 1)
    oracle.x(1)
elif target_item == 3:
    oracle.x(0)
    oracle.cz(0, 1)
    oracle.x(0)

# The Diffusion operator (inversion about the mean)
diffusion = QuantumCircuit(n_qubits, name="Diffusion")
diffusion.h(range(n_qubits))
diffusion.z(range(n_qubits))
diffusion.cz(0, 1)
diffusion.h(range(n_qubits))

# Number of Grover iterations (for a 4-item database, typically 1 iteration is optimal)
num_iterations = 1

# Apply the Grover iterations
for _ in range(num_iterations):
    qc.append(oracle, [i for i in range(n_qubits)])
    qc.append(diffusion, [i for i in range(n_qubits)])

# Measure the qubits
qc.measure(range(n_qubits), range(n_qubits))

# Simulate the circuit
simulator = Aer.get_backend('qasm_simulator')
compiled_circuit = transpile(qc, simulator)
job = simulator.run(compiled_circuit, shots=1024)
result = job.result()
counts = result.get_counts(qc)

# Print the results
print("Counts:", counts)
print(f"The most likely state is: {max(counts, key=counts.get)}")

# Optional: Plot the histogram
plot_histogram(counts)