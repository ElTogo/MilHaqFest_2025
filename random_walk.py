from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import plot_histogram

# Circuit pour une marche quantique sur un cycle de 4 nœuds (2 qubits position + 1 qubit coin)
qc = QuantumCircuit(3, 2)  # 3 qubits (q0,q1 position, q2 coin), 2 bits classiques pour lire la position

# Pas 1 de la marche quantique
qc.h(2)            # (a) Lancer de la pièce quantique (Hadamard sur le qubit coin)
# Opérateur de shift vers la gauche (coin = 1)
qc.cx(2, 0)        # Si coin (q2) = 1, basculer le LSB du registre position (q0)
qc.ccx(2, 0, 1)    # Si coin = 1 *et* LSB (q0) = 1, basculer le MSB (q1) – implémente l'addition modulaire (retrait) avec retenue
# Opérateur de shift vers la droite (coin = 0)
qc.x(2); qc.x(0)   # Préparation : on inverse coin et LSB pour traiter le cas coin=0 comme coin'=1
qc.cx(2, 0)        # (coin était 0 -> coin' = 1) Si coin original = 0, ceci flippe le LSB 
qc.ccx(2, 0, 1)    # Si coin original = 0 *et* LSB original = 1 (coin'=1 et LSB' après flip = 1), on flippe le MSB (carry)
qc.x(2); qc.x(0)   # Rétablir les registres (désinversion)

# Pas 2 de la marche quantique (on répète les mêmes opérations)
qc.h(2)            # Lancer à nouveau la pièce 
# Shift gauche (coin=1)
qc.cx(2, 0)
qc.ccx(2, 0, 1)
# Shift droit (coin=0)
qc.x(2); qc.x(0)
qc.cx(2, 0)
qc.ccx(2, 0, 1)
qc.x(2); qc.x(0)

qc.measure_all()

qc.draw('mpl', filename='quantum_walk_circuit.png')
simulator = AerSimulator()
sampler = Sampler(mode=simulator)

pass_manager = generate_preset_pass_manager(backend=simulator, optimization_level=1)
isa_circuit = pass_manager.run(qc)
job = sampler.run([isa_circuit], shots=100)
counts = job.result()[0].data.meas.get_counts()
plot_histogram(counts, filename="histogram.png")