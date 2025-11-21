import numpy as np
from scipy.linalg import expm  # pour le calcul de l'exponentielle de matrice
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

# Définir la matrice Hamiltonienne H pour un cycle à 4 nœuds
A = np.array([[0, 1, 0, 1],    # matrice d'adjacence du cycle 4
              [1, 0, 1, 0],
              [0, 1, 0, 1],
              [1, 0, 1, 0]], dtype=float)
deg = 2
L = A - deg*np.eye(4)         # Laplacien L = A - D (D=2I)
H = -1.0 * L                  # Hamiltonien H = -γ L avec γ=1
t = np.pi/2                   # temps d'évolution

# Calculer la matrice unitaire U = exp(-i H t)
U = expm(-1j * H * t)

# Créer un circuit quantique à 2 qubits (pour 4 états) et appliquer U
qc = QuantumCircuit(2, 2)
qc.append(Operator(U), [0, 1])   # appliquer l'opérateur unitaire sur les qubits 0 et 1
qc.measure([0, 1], [0, 1])      # mesure des deux qubits de position
