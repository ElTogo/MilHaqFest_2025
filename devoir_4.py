from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import plot_histogram


def build_n_hair_coulours_quantum_circuit(num_players: int) -> QuantumCircuit:
    circuit = QuantumCircuit(num_players*2)
    
    # état initial superposé des couleurs de cheveux
    for i in range(num_players):
        circuit.h(i)
    
    circuit.barrier()
    row_of_the_key = num_players


    for i in range(num_players - 1):
        circuit.cx(i + 1, row_of_the_key)
    circuit.barrier()

    # partage de l'information de la parité des couleurs des cheveux
    for i in range(num_players - 1):
        circuit.cx(row_of_the_key, row_of_the_key + i +1)
    circuit.barrier()

    # diffusion de l'information entre les joueurs
    for i in range(num_players -2):

        for j in range(num_players -2 - i):
            circuit.cx(i + 2 + j, row_of_the_key + i +1)
        circuit.barrier()

        for j in range(num_players -2 - i):
            circuit.cx(row_of_the_key + i + 1, row_of_the_key + i + j +2)

        circuit.barrier()

    circuit.measure_all()
    return circuit

num_players = 4
qc = build_n_hair_coulours_quantum_circuit(num_players)

simulator = AerSimulator()
sampler = Sampler(mode=simulator)

pass_manager = generate_preset_pass_manager(backend=simulator, optimization_level=1)
isa_circuit = pass_manager.run(qc)
job = sampler.run([isa_circuit], shots=10000)
counts = job.result()[0].data.meas.get_counts()
plot_histogram(counts, filename="histogram.png")
