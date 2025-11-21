from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
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
    # Lancer de la pièce quantique (Hadamard)
    qc.h(coin_qubit)
    
    # Shift conditionnel basé sur l'état du coin
    # Si coin=0: shift gauche, Si coin=1: shift droit
    
    # Shift gauche quand coin=1
    qc.cx(coin_qubit, pos_qubits[0])
    qc.ccx(coin_qubit, pos_qubits[0], pos_qubits[1])
    
    # Shift droit quand coin=0
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
    qc = QuantumCircuit(3, 2)  # 2 bits classiques pour les positions uniquement
    
    # Initialiser la position si différente de 0
    if initial_position & 1:  # bit 0
        qc.x(0)
    if initial_position & 2:  # bit 1
        qc.x(1)
    
    # Appliquer n pas de marche quantique
    for step in range(n_steps):
        quantum_walk_step(qc, coin_qubit=2, pos_qubits=[0, 1])
    
    # Mesurer UNIQUEMENT les qubits de position (ignorer le coin)
    qc.measure([0, 1], [0, 1])
    
    return qc

def simulate_quantum_walk(n_steps, shots=1000, initial_position=0):
    """
    Simule une marche quantique et retourne les résultats
    
    Args:
        n_steps: Nombre de pas
        shots: Nombre de mesures
        initial_position: Position initiale
    
    Returns:
        tuple: (circuit, counts_dict)
    """
    # Créer le circuit
    qc = create_quantum_walk(n_steps, initial_position)
    
    # Simuler avec AerSimulator
    simulator = AerSimulator()
    compiled_qc = transpile(qc, simulator)
    job = simulator.run(compiled_qc, shots=shots)
    result = job.result()
    counts_dict = result.get_counts()
    
    return qc, counts_dict

def format_position_results(counts):
    """
    Formate les résultats pour afficher les positions
    
    Args:
        counts: Dictionnaire des comptages
    
    Returns:
        dict: Positions formatées
    """
    positions = {}
    for bitstring, count in counts.items():
        pos = int(bitstring, 2)
        positions[f"Pos {pos}"] = count
    
    return positions

# Exemples d'utilisation
if __name__ == "__main__":
    print("="*60)
    print("MARCHE QUANTIQUE SUR CYCLE À 4 NŒUDS")
    print("="*60)
    
    # Tester différents nombres de pas
    steps_list = [1, 2, 4, 8]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, n_steps in enumerate(steps_list):
        print(f"\n--- Simulation avec {n_steps} pas ---")
        
        try:
            # Créer et simuler
            circuit, counts = simulate_quantum_walk(n_steps, shots=2000, initial_position=0)
            
            # Formater les positions
            positions = format_position_results(counts)
            
            print(f"Profondeur du circuit: {circuit.depth()}")
            print(f"Nombre d'états mesurés: {len(positions)}")
            print(f"Distribution: {positions}")
            
            # Tracer l'histogramme
            ax = axes[idx]
            colors = ['steelblue' if int(k.split()[1]) < 4 else 'red' for k in positions.keys()]
            ax.bar(positions.keys(), positions.values(), color=colors, alpha=0.7)
            ax.set_title(f'{n_steps} pas de marche quantique')
            ax.set_xlabel('Position (0-3)')
            ax.set_ylabel('Nombre de mesures')
            ax.set_ylim([0, 2000])
            ax.grid(True, alpha=0.3, axis='y')
        
        except Exception as e:
            print(f"Erreur lors de la simulation: {e}")
            ax = axes[idx]
            ax.text(0.5, 0.5, f'Erreur: {str(e)}', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('quantum_walk_comparison.png', dpi=300, bbox_inches='tight')
    print("\nGraphique sauvegardé dans 'quantum_walk_comparison.png'")
    plt.show()
    
    # Simulation détaillée pour 4 pas
    print("\n" + "="*60)
    print("SIMULATION DÉTAILLÉE: 4 PAS")
    print("="*60)
    try:
        circuit_4, counts_4 = simulate_quantum_walk(4, shots=5000)
        positions_4 = format_position_results(counts_4)
        
        print(f"Distribution finale: {positions_4}")
        
        # Afficher l'histogramme
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(positions_4.keys(), positions_4.values(), color='steelblue', alpha=0.7)
        ax.set_title('Distribution des positions après 4 pas')
        ax.set_xlabel('Position (0-3)')
        ax.set_ylabel('Nombre de mesures')
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig("histogram.png", dpi=300, bbox_inches='tight')
        print("Histogramme sauvegardé dans 'histogram.png'")
        plt.show()
        
        # Dessiner le circuit
        circuit_4.draw('mpl', filename='quantum_walk_4steps_circuit.png', fold=80)
        print("Circuit sauvegardé dans 'quantum_walk_4steps_circuit.png'")
    
    except Exception as e:
        print(f"Erreur: {e}")
    
    print("\n" + "="*60)
    print("SIMULATION TERMINÉE!")
    print("="*60)