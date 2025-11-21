from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def quantum_walk_step(qc, coin_qubit, pos_qubits):
    """
    Applique un pas de marche quantique sur un cycle à 4 nœuds
    
    Args:
        qc: Le circuit quantique
        coin_qubit: Index du qubit coin
        pos_qubits: Liste des indices des qubits de position [LSB, MSB]
    """
    # Lancer de la pièce quantique
    qc.h(coin_qubit)
    
    # Increment modulo 4 (shift gauche, quand coin=1)
    # 00 -> 01, 01 -> 10, 10 -> 11, 11 -> 00
    qc.cx(coin_qubit, pos_qubits[0])
    qc.ccx(coin_qubit, pos_qubits[0], pos_qubits[1])
    
    # Decrement modulo 4 (shift droit, quand coin=0)
    # 00 -> 11, 01 -> 00, 10 -> 01, 11 -> 10
    qc.x(coin_qubit)
    qc.ccx(coin_qubit, pos_qubits[0], pos_qubits[1])
    qc.cx(coin_qubit, pos_qubits[0])
    qc.x(coin_qubit)

def create_quantum_walk(n_steps, initial_position=0):
    """
    Crée un circuit de marche quantique avec n pas
    
    Args:
        n_steps: Nombre de pas de marche quantique
        initial_position: Position initiale (0-3 pour un cycle de 4 nœuds)
    
    Returns:
        QuantumCircuit: Le circuit quantique complet
    """
    # 3 qubits: q0, q1 pour position, q2 pour coin
    # 3 bits classiques pour mesurer tous les qubits
    qc = QuantumCircuit(3, 3)
    
    # Initialiser la position si différente de 0
    if initial_position & 1:  # bit 0
        qc.x(0)
    if initial_position & 2:  # bit 1
        qc.x(1)
    
    # Appliquer n pas de marche quantique
    for step in range(n_steps):
        quantum_walk_step(qc, coin_qubit=2, pos_qubits=[0, 1])
        qc.barrier()  # Pour visualisation
    
    # Mesurer TOUS les qubits (position + coin)
    qc.measure([0, 1, 2], [0, 1, 2])
    
    return qc

def simulate_quantum_walk(n_steps, shots=1000, initial_position=0):
    """
    Simule une marche quantique et retourne les résultats
    
    Args:
        n_steps: Nombre de pas
        shots: Nombre de mesures
        initial_position: Position initiale
    
    Returns:
        dict: Dictionnaire des comptages
    """
    # Créer le circuit
    qc = create_quantum_walk(n_steps, initial_position)
    
    # Simuler avec AerSimulator et Sampler
    simulator = AerSimulator()
    sampler = Sampler(mode=simulator)
    
    pass_manager = generate_preset_pass_manager(backend=simulator, optimization_level=1)
    isa_circuit = pass_manager.run(qc)
    job = sampler.run([isa_circuit], shots=shots)
    
    # Récupérer les résultats
    result = job.result()[0]
    # Obtenir les counts depuis le DataBin
    counts_dict = result.data.c.get_counts()
    
    return qc, counts_dict

def extract_position_distribution(counts):
    """
    Extrait la distribution des positions en ignorant l'état du coin
    
    Args:
        counts: Dictionnaire des comptages (3 bits)
    
    Returns:
        dict: Distribution des positions uniquement
    """
    position_counts = {}
    
    for bitstring, count in counts.items():
        # Extraire les 2 premiers bits (position), ignorer le 3ème (coin)
        position_bits = bitstring[:2]  # Les 2 bits de position
        
        if position_bits in position_counts:
            position_counts[position_bits] += count
        else:
            position_counts[position_bits] = count
    
    return position_counts

# Exemples d'utilisation
if __name__ == "__main__":
    print("="*60)
    print("MARCHE QUANTIQUE AVEC N ÉTAPES")
    print("="*60)
    
    # Tester différents nombres de pas
    steps_list = [2, 4, 8, 16]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, n_steps in enumerate(steps_list):
        print(f"\n--- Simulation avec {n_steps} pas ---")
        
        # Créer et simuler
        circuit, counts = simulate_quantum_walk(n_steps, shots=2000, initial_position=0)
        
        # Extraire uniquement les positions
        position_counts = extract_position_distribution(counts)
        
        print(f"Nombre de portes: {circuit.depth()}")
        print(f"Distribution des positions: {position_counts}")
        
        # Tracer l'histogramme
        ax = axes[idx]
        
        # Convertir les bits en positions
        positions = {}
        for bitstring, count in position_counts.items():
            pos = int(bitstring, 2)
            positions[f"Pos {pos}"] = count
        
        ax.bar(positions.keys(), positions.values())
        ax.set_title(f'{n_steps} pas de marche quantique')
        ax.set_xlabel('Position')
        ax.set_ylabel('Nombre de mesures')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quantum_walk_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Dessiner le circuit pour 4 pas
    print("\n--- Circuit pour 4 pas ---")
    circuit_4, counts_4 = simulate_quantum_walk(4, shots=10000)
    
    # Extraire les positions
    position_counts_4 = extract_position_distribution(counts_4)
    
    # Sauvegarder l'histogramme
    plot_histogram(position_counts_4, filename="histogram.png")
    print("Histogramme sauvegardé dans 'histogram.png'")
    
    # Dessiner le circuit
    circuit_4.draw('mpl', filename='quantum_walk_4steps_circuit.png')
    print("Circuit sauvegardé dans 'quantum_walk_4steps_circuit.png'")
    
    print("\n" + "="*60)
    print("SIMULATION TERMINÉE!")
    print("="*60)