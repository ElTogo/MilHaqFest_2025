# %%
import numpy as np
from math import sqrt, pi, exp, log, erf
from qiskit import QuantumCircuit
from qiskit_aer import Aer
import scipy

# %%
def black_scholes_quantum_mc(
        S0,
        K,
        r,
        sigma,
        T,
        n_qubits = 3,
        shots = 10_000,
        option_type= "put",
        n_std = 3.0,
    ):
    """
    black_scholes_quantum_mc apprioximate the call and put values by runing a monte carlo
    simulation and
    """

    # 1) Paramètres de la loi normale de log S_T
    mean = (r - 0.5 * sigma**2) * T
    std = sigma * sqrt(T)

    # 2) Discrétisation de z sur [mean - n_std*std, mean + n_std*std]
    n_points = 2**n_qubits
    z_min = mean - n_std * std
    z_max = mean + n_std * std
    z_vals = np.linspace(z_min, z_max, n_points)

    # 3) Probas approximatives sur la grille : densité normale
    pdf_vals = (
        1.0
        / (std * sqrt(2.0 * pi))
        * np.exp(-0.5 * ((z_vals - mean) / std) ** 2)
    )
    p_vals = pdf_vals / pdf_vals.sum()  # normalisation

    # 4) Amplitudes quantiques = sqrt(probabilités)
    amplitudes = np.sqrt(p_vals)
    amplitudes = amplitudes / np.linalg.norm(amplitudes)  # sécurité de normalisation

    # 5) Circuit quantique : initialise la superposition puis mesure
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.initialize(amplitudes, list(range(n_qubits)))
    qc.measure_all()

    # 6) Simulation sur Aer
    backend = Aer.get_backend("aer_simulator")
    job = backend.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # 7) Calcul des payoffs associés à chaque z_i
    ST_vals = S0 * np.exp(z_vals)

    if option_type.lower() == "put":
        payoff_grid = np.maximum(K - ST_vals, 0.0)
    elif option_type.lower() == "call":
        payoff_grid = np.maximum(ST_vals - K, 0.0)
    else:
        raise ValueError("option_type doit être 'put' ou 'call'.")

    # 8) Espérance du payoff en utilisant les résultats de mesure
    total_shots = sum(counts.values())
    exp_payoff = 0.0

    for bitstring, c in counts.items():
        # Gérer le cas avec plusieurs registres classiques (espaces dans la clé)
        bit_clean = bitstring.replace(" ", "")
        # Qiskit est big-endian, on renverse pour indexer la grille en little-endian
        idx = int(bit_clean[::-1], 2)
        exp_payoff += payoff_grid[idx] * c

    exp_payoff /= total_shots

    # 9) Actualisation pour obtenir le prix
    price = exp(-r * T) * exp_payoff
    return float(price)


# ======================
#  Exemple d'utilisation
# ======================
# %%
if __name__ == "__main__":
    # ====== INPUTS UTILISATEUR ======
    # Tu modifies juste ces lignes et tu relances le script / la cellule

    S0 = 100.0      # prix spot
    K = 100.0       # strike
    r = 0.05        # taux sans risque
    sigma = 0.2     # volatilité
    t = 0.0         # temps actuel
    T = 1.0         # maturité
    option_type = "put"   # "put" ou "call"

    n_qubits = 3          # 2**n_qubits points de discrétisation
    shots = 10_000        # nombre de mesures (samples quantiques)
    n_sim_classique = 100_000  # pour MC classique

    # temps à l'échéance
    tau = T - t

    # Monte Carlo "quantique"
    mc_quantum = black_scholes_quantum_mc(
        S0, K, r, sigma, tau,
        n_qubits=n_qubits,
        shots=shots,
        option_type=option_type,
    )

    print(f"=== Inputs ===")
    print(f"S0 = {S0}, K = {K}, r = {r}, sigma = {sigma}, t = {t}, T = {T}, type = {option_type}")
    print(f"n_qubits = {n_qubits}, shots = {shots}")
    print()
    print(f"Monte Carlo 'quantique' (Qiskit) : {mc_quantum:.4f}")

# %%
