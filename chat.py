# Import des bibliothèques nécessaires
import numpy as np
import networkx as nx
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

# Paramètres (modifiable par l'utilisateur)
n_nodes = 5        # nombre de nœuds du graphe
p_edge = 0.5       # probabilité d'apparition d'une arête (modèle Erdős–Rényi)
n_steps = 3        # nombre de pas de la marche quantique
initial_node = 2   # nœud de départ du marcheur quantique

# 1. Générer un graphe aléatoire binomial G(n_nodes, p_edge)
G = nx.erdos_renyi_graph(n_nodes, p_edge)

# 2. Déterminer le nombre de qubits requis pour encoder les positions des nœuds en binaire
if n_nodes > 1:
    p_qubits = int(np.ceil(np.log2(n_nodes)))
else:
    p_qubits = 1  # au moins 1 qubit si un seul nœud
c_qubits = p_qubits  # on utilise le même nombre de qubits pour la pièce quantique (coin)

# Créer les registres quantiques pour la position, la pièce, et un ancilla pour l'opérateur shift
pos = QuantumRegister(p_qubits, name='pos')      # registre position (nœud)
coin = QuantumRegister(c_qubits, name='coin')    # registre pièce (coin qubit)
anc = QuantumRegister(1, name='ancilla')         # qubit ancilla pour les opérations de swap contrôlé
creg = ClassicalRegister(p_qubits, name='c_out') # registre classique pour lire la position finale
qc = QuantumCircuit(pos, coin, anc, creg)

# Initialiser le marcheur sur le nœud de départ (initial_node)
# Par défaut, tous les qubits commencent en |0..0>, donc si initial_node = 0, on ne fait rien.
if initial_node >= n_nodes:
    raise ValueError("Initial node index is out of range.")
if initial_node != 0:
    # Préparer la représentation binaire du nœud de départ et initialiser les qubits de position
    init_bin = format(initial_node, 'b').zfill(p_qubits)  # chaîne binaire sur p_qubits bits
    # Remarque : pos[0] correspond au bit de poids faible (LSB) de la position
    for i, bit in enumerate(reversed(init_bin)):
        if bit == '1':
            qc.x(pos[i])  # met le qubit à 1 si le bit correspondant dans initial_node est 1

# 3. Construire le circuit de la marche quantique sur n_steps pas
for step in range(n_steps):
    # Opérateur coin : appliquer un Hadamard sur chaque qubit de pièce pour superposer les deux (ou 2^c_qubits) états
    for qb in range(c_qubits):
        qc.h(coin[qb])
    
    # Opérateur shift : déplacer le marcheur conditionnellement selon l'état de la pièce et les arêtes du graphe
    # Pour chaque arête (u, v) du graphe, on effectue un SWAP contrôlé entre les registres position et coin
    for (u, v) in G.edges():
        if u == v:
            # Ignorer les boucles (auto-arêtes) si présentes (pas attendues dans Erdős–Rényi simple)
            continue
        # Calculer l'état de l'ancilla = 1 si (position = u ET coin = v) OU (position = v ET coin = u)
        # Pattern 1 : position == u et coin == v
        bin_u = format(u, 'b').zfill(p_qubits)
        bin_v = format(v, 'b').zfill(p_qubits)
        # Appliquer X sur les qubits dont le bit correspondant de u ou v vaut 0 (pour préparer la condition)
        for i, bit in enumerate(reversed(bin_u)):
            if bit == '0':
                qc.x(pos[i])
        for j, bit in enumerate(reversed(bin_v)):
            if bit == '0':
                qc.x(coin[j])
        # NOT multicontrollé (mcx) : toutes les qubits de pos+coin en contrôle, cible = ancilla
        qc.mcx(list(pos) + list(coin), anc[0])
        # Annuler les X appliqués (remettre les registres pos et coin comme avant)
        for j, bit in enumerate(reversed(bin_v)):
            if bit == '0':
                qc.x(coin[j])
        for i, bit in enumerate(reversed(bin_u)):
            if bit == '0':
                qc.x(pos[i])
        
        # Pattern 2 : position == v et coin == u (même procédé en inversant u et v)
        for i, bit in enumerate(reversed(bin_v)):
            if bit == '0':
                qc.x(pos[i])
        for j, bit in enumerate(reversed(bin_u)):
            if bit == '0':
                qc.x(coin[j])
        qc.mcx(list(pos) + list(coin), anc[0])
        # Annuler les X pour le pattern 2
        for j, bit in enumerate(reversed(bin_u)):
            if bit == '0':
                qc.x(coin[j])
        for i, bit in enumerate(reversed(bin_v)):
            if bit == '0':
                qc.x(pos[i])
        
        # À ce stade, l'ancilla vaut 1 si l'état |pos, coin> correspond à |u, v> ou |v, u>
        # Effectuer le SWAP contrôlé (Fredkin) entre les registres position et coin en utilisant l'ancilla
        for b in range(p_qubits):
            qc.cswap(anc[0], pos[b], coin[b])
        
        # Uncompute de l'ancilla (remettre l'ancilla à 0 en inversant les opérations de contrôle précédentes)
        # Pattern 2 (inversion)
        for i, bit in enumerate(reversed(bin_v)):
            if bit == '0':
                qc.x(pos[i])
        for j, bit in enumerate(reversed(bin_u)):
            if bit == '0':
                qc.x(coin[j])
        qc.mcx(list(pos) + list(coin), anc[0])
        # Annuler les X du pattern 2
        for j, bit in enumerate(reversed(bin_u)):
            if bit == '0':
                qc.x(coin[j])
        for i, bit in enumerate(reversed(bin_v)):
            if bit == '0':
                qc.x(pos[i])
        # Pattern 1 (inversion)
        for i, bit in enumerate(reversed(bin_u)):
            if bit == '0':
                qc.x(pos[i])
        for j, bit in enumerate(reversed(bin_v)):
            if bit == '0':
                qc.x(coin[j])
        qc.mcx(list(pos) + list(coin), anc[0])
        # Annuler les X du pattern 1
        for j, bit in enumerate(reversed(bin_v)):
            if bit == '0':
                qc.x(coin[j])
        for i, bit in enumerate(reversed(bin_u)):
            if bit == '0':
                qc.x(pos[i])
    # Fin du parcours de chaque arête
# Fin du parcours de chaque pas

# 4. Mesurer le registre de position pour obtenir la distribution finale des positions
qc.measure(pos, creg)

# 5. Simuler le circuit sur le simulateur Qiskit Aer
simulator = AerSimulator()
shots = 10000  # nombre de mesures pour estimer la distribution
# Transpiler et exécuter
transpiled_qc = transpile(qc, simulator)
job = simulator.run(transpiled_qc, shots=shots)
result = job.result()
counts = result.get_counts()

# 6. Calculer et afficher la distribution de probabilité finale sur les positions
probabilities = {int(outcome, 2): count/shots for outcome, count in counts.items()}
# Ajouter les positions non présentes dans counts avec probabilité 0
for node in range(n_nodes):
    probabilities.setdefault(node, 0.0)

# Affichage de la distribution de probabilité (histogramme)
positions = sorted(probabilities.keys())
probs = [probabilities[node] for node in positions]

print(f"\nGraphe généré: {n_nodes} nœuds, {G.number_of_edges()} arêtes")
print(f"Arêtes: {list(G.edges())}")
print(f"\nDistribution finale après {n_steps} pas depuis le nœud {initial_node}:")
for node in positions:
    print(f"  Nœud {node}: {probabilities[node]:.4f}")

plt.figure(figsize=(8,5))
plt.bar(positions, probs, tick_label=positions, color='steelblue', alpha=0.7)
plt.xlabel("Position (nœud)")
plt.ylabel("Probabilité")
plt.title(f"Distribution finale sur les positions après {n_steps} pas")
plt.grid(True, alpha=0.3, axis='y')
plt.savefig('quantum_walk_random_graph.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nGraphique sauvegardé dans 'quantum_walk_random_graph.png'")
