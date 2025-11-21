# imports
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

# number of steps
steps = 4
# qubits needed for position encoding (2*steps+1 positions)
n_qubits = int(np.ceil(np.log2(2*steps + 1)))
# additional qubit for coin
total_qubits = n_qubits + 1

def create_quantum_walk_circuit(steps, initial_position=0):
    """Create a quantum walk circuit with coin operator"""
    qr = QuantumRegister(total_qubits, 'q')
    cr = ClassicalRegister(total_qubits, 'c')
    qc = QuantumCircuit(qr, cr)
    
    # Initialize coin qubit in superposition
    qc.h(0)
    
    # Initialize position (middle position)
    middle = steps
    for i, bit in enumerate(format(middle, f'0{n_qubits}b')[::-1]):
        if bit == '1':
            qc.x(i+1)
    
    # Apply quantum walk steps
    for step in range(steps):
        # Coin operator (Hadamard on coin qubit)
        qc.h(0)
        
        # Conditional shift operator
        # If coin is |0⟩, decrement position
        # If coin is |1⟩, increment position
        
        # Increment (controlled by coin qubit = 1)
        for i in range(n_qubits):
            controls = [0] + [j+1 for j in range(i)]
            qc.mcx(controls, i+1)
        
        # Decrement (controlled by coin qubit = 0)
        qc.x(0)
        for i in range(n_qubits):
            controls = [0] + [j+1 for j in range(i)]
            qc.mcx(controls, i+1)
        qc.x(0)
        
        qc.barrier()
    
    # Measure all qubits
    qc.measure(qr, cr)
    
    return qc

# Create the quantum walk circuit
circuit = create_quantum_walk_circuit(steps)

# Display the circuit
print(circuit)

# Simulate the circuit
simulator = AerSimulator()
job = simulator.run(circuit, shots=8192)
result = job.result()
counts = result.get_counts()

# Process results
mode_to_walk_pos_mapping = {
    0: 4,
    1: 2,
    2: 2,
    3: 0,
    4: 0,
    5: -2,
    6: -2,
    7: -4
}

# Calculate probabilities
total_shots = sum(counts.values())
walk_pos = range(-steps, steps+1)
walk_probs = [0]*(2*steps+1)

# Map measurement results to walk positions
for bitstring, count in counts.items():
    # Extract position from measurement (ignore coin qubit)
    position_bits = bitstring[:-1]
    position = int(position_bits, 2)
    
    # Convert to walk position
    walk_position = position - steps
    if walk_position in walk_pos:
        idx = walk_position + steps
        walk_probs[idx] += count / total_shots

# Print results
print("\nWalk positions and probabilities:")
for w_p, w_p_p in zip(walk_pos, walk_probs):
    print(f"Walk position: {w_p}, Probability: {w_p_p:.4f}")

# Plot the walk positions distribution
plt.figure(figsize=(10, 6))
plt.bar(walk_pos, walk_probs)
plt.xticks(walk_pos)
plt.xlabel("Position")
plt.ylabel("Probability")
plt.title("Quantum Walk Position Distribution")
plt.grid(True, alpha=0.3)
plt.show()