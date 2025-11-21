# Import des bibliothèques nécessaires
import numpy as np
from qiskit.quantum_info import Kraus, DensityMatrix  # Qiskit pour définir le canal quantique et manipuler la densité
import matplotlib.pyplot as plt

# Paramètres de simulation
gamma = 0.2             # taux de dissipation par étape (probabilité de perdre l'excitation à chaque interaction)
n_steps = 15            # nombre d'étapes de la simulation (durée du mouvement brownien quantique simulé)

# Initialisation de l'état initial du qubit (ici |1>, état excité)
initial_state = DensityMatrix.from_label('1')  # matrice de densité initiale = |1><1|

# Définition des opérateurs de Kraus pour le canal d'amortissement d'amplitude
K0 = np.array([[1, 0],
               [0, np.sqrt(1 - gamma)]])   # K0 conserve l'état (|0> reste |0>, |1> reste |1> avec sqrt(1-gamma))
K1 = np.array([[0, np.sqrt(gamma)],
               [0, 0]])                   # K1 fait la transition |1> -> |0> avec amplitude sqrt(gamma)
channel = Kraus([K0, K1])                 # Création de l'objet canal quantique Kraus dans Qiskit

# Boucle de simulation sur chaque étape
state = initial_state.copy()    # copie de la matrice de densité initiale
p_excited = []                  # liste pour stocker la probabilité d'être en |1> après chaque étape
for step in range(n_steps + 1):
    # Calcul de la probabilité d'être dans |1> à la current étape
    probs = state.probabilities()       # distribution de probas dans la base calcul (|0>, |1>)
    p_excited.append(probs[1])          # probabilité d'être en |1> (état excité)
    # Appliquer le canal d'amortissement pour passer à l'étape suivante (sauf après la dernière itération)
    if step < n_steps:
        state = state.evolve(channel)

# (Optionnel) Affichage des probabilités à chaque étape
print("Probabilités d'être en |1> à chaque étape :", p_excited)

# Tracé de la trajectoire de la population |1> au cours du temps
plt.figure(figsize=(6,4))
plt.plot(range(n_steps + 1), p_excited, marker='o', color='orange')
plt.title("Probabilité d'état excité du qubit vs nombre d'étapes")
plt.xlabel("Étape")
plt.ylabel("Probabilité d'être en |1>")
plt.grid(True)
plt.show()
