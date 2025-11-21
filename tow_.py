# imports
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from mpl_toolkits.mplot3d import Axes3D

# number of steps
steps = 4
# qubits needed for position encoding (2*steps+1 positions)
n_qubits = int(np.ceil(np.log2(2*steps + 1)))
# additional qubit for coin
total_qubits = n_qubits + 1

def create_quantum_walk_circuit(steps, initial_position=None):
    """Create a quantum walk circuit with coin operator"""
    qr = QuantumRegister(total_qubits, 'q')
    cr = ClassicalRegister(total_qubits, 'c')
    qc = QuantumCircuit(qr, cr)
    
    # Initialize coin qubit in superposition
    qc.h(0)
    
    # Initialize position (middle position if not specified)
    if initial_position is None:
        initial_position = steps
    
    for i, bit in enumerate(format(initial_position, f'0{n_qubits}b')[::-1]):
        if bit == '1':
            qc.x(i+1)
    
    # Apply quantum walk steps
    for step in range(steps):
        # Coin operator (Hadamard on coin qubit)
        qc.h(0)
        
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

# dictionary to map the mode to the position
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

print("="*50)
print("SINGLE PHOTON / SINGLE WALKER SIMULATION")
print("="*50)

# Create the quantum walk circuit for single walker
circuit = create_quantum_walk_circuit(steps, initial_position=3)

# Display the circuit
print("\nQuantum Circuit:")
print(circuit)

# Simulate the circuit
simulator = AerSimulator()
job = simulator.run(circuit, shots=8192)
result = job.result()
counts = result.get_counts()

print("\nOutput distribution:")
for bitstring, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{bitstring}: {count/8192:.4f}")

# Calculate probabilities
total_shots = sum(counts.values())
walk_pos = list(range(-steps, steps+1))
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
plt.title("Single Quantum Walker - Position Distribution")
plt.grid(True, alpha=0.3)
plt.show()

print("\n" + "="*50)
print("TWO PHOTONS / TWO WALKERS SIMULATION")
print("="*50)

def create_two_walker_circuit(steps):
    """Create a quantum walk circuit for two walkers"""
    # Each walker needs position qubits + coin qubit
    walker1_qubits = n_qubits + 1
    walker2_qubits = n_qubits + 1
    total_qubits_2w = walker1_qubits + walker2_qubits
    
    qr = QuantumRegister(total_qubits_2w, 'q')
    cr = ClassicalRegister(total_qubits_2w, 'c')
    qc = QuantumCircuit(qr, cr)
    
    # Initialize both coin qubits in superposition
    qc.h(0)  # Walker 1 coin
    qc.h(walker1_qubits)  # Walker 2 coin
    
    # Initialize positions (walker 1 at position 3, walker 2 at position 4)
    for i, bit in enumerate(format(3, f'0{n_qubits}b')[::-1]):
        if bit == '1':
            qc.x(i+1)
    
    for i, bit in enumerate(format(4, f'0{n_qubits}b')[::-1]):
        if bit == '1':
            qc.x(walker1_qubits + i + 1)
    
    # Apply quantum walk steps for both walkers
    for step in range(steps):
        # Coin operators
        qc.h(0)
        qc.h(walker1_qubits)
        
        # Walker 1 increment/decrement
        for i in range(n_qubits):
            controls = [0] + [j+1 for j in range(i)]
            qc.mcx(controls, i+1)
        qc.x(0)
        for i in range(n_qubits):
            controls = [0] + [j+1 for j in range(i)]
            qc.mcx(controls, i+1)
        qc.x(0)
        
        # Walker 2 increment/decrement
        for i in range(n_qubits):
            controls = [walker1_qubits] + [walker1_qubits + j + 1 for j in range(i)]
            qc.mcx(controls, walker1_qubits + i + 1)
        qc.x(walker1_qubits)
        for i in range(n_qubits):
            controls = [walker1_qubits] + [walker1_qubits + j + 1 for j in range(i)]
            qc.mcx(controls, walker1_qubits + i + 1)
        qc.x(walker1_qubits)
        
        qc.barrier()
    
    qc.measure(qr, cr)
    return qc, walker1_qubits

# Create two-walker circuit
circuit_2w, split_point = create_two_walker_circuit(steps)

print("\nTwo-Walker Quantum Circuit created")

# Simulate
job_2w = simulator.run(circuit_2w, shots=8192)
result_2w = job_2w.result()
counts_2w = result_2w.get_counts()

print("\nOutput distribution:")
for bitstring, count in sorted(counts_2w.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{bitstring}: {count/8192:.4f}")

# Process two-walker results
walk_probs_2d = np.zeros((2*steps+1, 2*steps+1))

for bitstring, count in counts_2w.items():
    # Split bitstring into two walkers
    walker2_bits = bitstring[:split_point-1]
    walker1_bits = bitstring[split_point:-1]
    
    pos1 = int(walker1_bits, 2) - steps
    pos2 = int(walker2_bits, 2) - steps
    
    if pos1 in walk_pos and pos2 in walk_pos:
        idx1 = pos1 + steps
        idx2 = pos2 + steps
        walk_probs_2d[idx1, idx2] += count / 8192

# Plot 3D bar chart
x, y = np.meshgrid(walk_pos, walk_pos)
cmap = plt.get_cmap('jet')
max_height = np.max(walk_probs_2d.flatten())
min_height = np.min(walk_probs_2d.flatten())
rgba = [cmap((k-min_height)/max_height) if k > 0 else (0,0,0,0) for k in walk_probs_2d.flatten()]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.bar3d(x.flatten(), y.flatten(), np.zeros((2*steps+1)*(2*steps+1)), 
         1, 1, walk_probs_2d.flatten(), color=rgba)
ax.set_xlabel("Walker 1 Position")
ax.set_ylabel("Walker 2 Position")
ax.set_zlabel("Probability")
ax.set_title("Two Quantum Walkers - Joint Position Distribution")
plt.show()

print("\nSimulation completed!")