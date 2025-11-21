import time

import numpy as np
import pandas as pd
from math import sqrt, pi, exp, log, erf
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit import transpile

MSE_ERROR_SUM = 0

def normal_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def black_scholes_analytical(S0, K, r, sigma, T, option_type="call"):
    """
    Formule analytique de Black-Scholes
    """
    if T <= 0:
        if option_type == "call":
            return max(S0 - K, 0)
        else:
            return max(K - S0, 0)
    
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    
    if option_type == "call":
        price = S0 * normal_cdf(d1) - K * exp(-r * T) * normal_cdf(d2)
    else:  # put
        price = K * exp(-r * T) * normal_cdf(-d2) - S0 * normal_cdf(-d1)
    
    return price

def black_scholes_quantum_mc(
    S0,
    K,
    r,
    sigma,
    T,
    n_qubits = 4,
    shots = 10_000,
    option_type = "put",
    n_std = 3.0,
):
    """
    Monte Carlo "quantique" pour prix d'option européenne sous Black-Scholes.
    Version corrigée avec AerSimulator
    """
    # 1) Paramètres de la loi normale de log S_T
    mean = log(S0) + (r - 0.5 * sigma**2) * T
    std = sigma * sqrt(T)

    # 2) Discrétisation de log(S_T) sur [mean - n_std*std, mean + n_std*std]
    n_points = 2**n_qubits
    z_min = mean - n_std * std
    z_max = mean + n_std * std
    z_vals = np.linspace(z_min, z_max, n_points)

    # 3) Probas approximatives sur la grille : densité log-normale
    pdf_vals = (
        1.0
        / (std * sqrt(2.0 * pi))
        * np.exp(-0.5 * ((z_vals - mean) / std) ** 2)
    )
    p_vals = pdf_vals / pdf_vals.sum()  # normalisation

    # 4) Amplitudes quantiques = sqrt(probabilités)
    amplitudes = np.sqrt(p_vals)
    amplitudes = amplitudes / np.linalg.norm(amplitudes)

    # 5) Circuit quantique : initialise la superposition puis mesure
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.initialize(amplitudes, list(range(n_qubits)))
    qc.measure(list(range(n_qubits)), list(range(n_qubits)))

    # 6) Simulation sur AerSimulator (nouvelle API)
    simulator = AerSimulator()
    transpiled_qc = transpile(qc, simulator)
    job = simulator.run(transpiled_qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # 7) Calcul des payoffs associés à chaque z_i
    ST_vals = np.exp(z_vals)  # z_vals est déjà log(S_T)

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
        # Gérer le cas avec plusieurs registres classiques
        bit_clean = bitstring.replace(" ", "")
        # Indexer la grille
        idx = int(bit_clean, 2)
        if idx < len(payoff_grid):
            exp_payoff += payoff_grid[idx] * c

    exp_payoff /= total_shots

    # 9) Actualisation pour obtenir le prix
    price = exp(-r * T) * exp_payoff
    return float(price)

def process_csv_and_predict(csv_file, output_file, method="analytical"):
    """
    Lit le CSV, calcule les valeurs Put et Call, et sauvegarde les résultats
    
    Args:
        csv_file: chemin vers le fichier CSV d'entrée
        output_file: chemin pour sauvegarder les résultats
        method: "analytical" ou "quantum"
    """
    # Lire le CSV avec le bon séparateur (virgule)
    df = pd.read_csv(csv_file, sep=',')
    
    # Remplacer les virgules par des points pour les nombres
    for col in df.columns:
        if col != 'Put Value Black-Scholes Model' and col != 'Call Value Black-Scholes Model':
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
    
    # Calculer Put et Call pour chaque ligne
    put_values = []
    call_values = []
    
    print(f"Calcul des options avec la méthode: {method}")
    print("="*80)
    
    for idx, row in df.iterrows():
        r = row['r']
        K = row['K']
        S0 = row['S0']
        sigma = row['Sigma']
        T = row['T']
        
        print(f"\nLigne {idx+1}: S0={S0:.3f}, K={K:.3f}, r={r:.3f}, σ={sigma:.3f}, T={T:.1f}")
        
        if method == "analytical":
            # Formule analytique (plus rapide et précise)
            put_val = black_scholes_analytical(S0, K, r, sigma, T, option_type="put")
            call_val = black_scholes_analytical(S0, K, r, sigma, T, option_type="call")
        else:
            # Monte Carlo quantique
            put_val = black_scholes_quantum_mc(S0, K, r, sigma, T, 
                                              n_qubits=4, shots=20000, 
                                              option_type="put")
            call_val = black_scholes_quantum_mc(S0, K, r, sigma, T, 
                                               n_qubits=4, shots=20000, 
                                               option_type="call")
        
        put_values.append(put_val)
        call_values.append(call_val)
        
        print(f"  Put = {put_val:.6f}, Call = {call_val:.6f}")
    
    # Ajouter les colonnes calculées
    df['Put Value Black-Scholes Model'] = put_values
    df['Call Value Black-Scholes Model'] = call_values
    
    # Sauvegarder avec des virgules comme séparateur décimal pour Excel
    df_output = df.copy()
    for col in ['Put Value Black-Scholes Model', 'Call Value Black-Scholes Model']:
        df_output[col] = df_output[col].apply(lambda x: f"{x:.6f}".replace('.', ','))
    
    df_output.to_csv(output_file, sep=',', index=False)
    
    print("\n" + "="*80)
    print(f"Résultats sauvegardés dans: {output_file}")
    print("="*80)
    
    return df

if __name__ == "__main__":
    # Input/output data
    input_file = r"input data.csv"
    output_analytical = "results_black_scholes_analytical.csv"

    print("\n### MÉTHODE ANALYTIQUE BLACK-SCHOLES ###\n")
    t = time.time()
    df_analytical = process_csv_and_predict(input_file, output_analytical, method="analytical")
    print(f"Analytical performance : {time.time() - t} seconds")

    print("\n### MÉTHODE MONTE CARLO QUANTIQUE ###\n")
    output_quantum = "results_black_scholes_quantum.csv"
    t = time.time()
    df_quantum = process_csv_and_predict(input_file, output_quantum, method="quantum")
    print(f"Quantum performance : {time.time() - t} seconds")

    # Comparaison
    print("\n" + "="*80)
    print("COMPARAISON DES DEUX MÉTHODES")
    print("="*80)
    comparison = pd.DataFrame({
        'Ligne': range(1, len(df_analytical) + 1),
        'Put Analytique': df_analytical['Put Value Black-Scholes Model'],
        'Put Quantique': df_quantum['Put Value Black-Scholes Model'],
        'Call Analytique': df_analytical['Call Value Black-Scholes Model'],
        'Call Quantique': df_quantum['Call Value Black-Scholes Model'],
    })
    mse_error_put = 0
    mse_error_call = 0
    for i in range(len(df_analytical)):
        mse_error_put += pow(df_analytical['Put Value Black-Scholes Model'][i]
                         - df_quantum['Put Value Black-Scholes Model'][i], 2)
        mse_error_call += pow(df_analytical['Call Value Black-Scholes Model'][i]
                              - df_quantum['Call Value Black-Scholes Model'][i], 2)
    mse_error_put = mse_error_put / len(df_analytical)
    mse_error_call = mse_error_call / len(df_quantum)
    print(f"MSE ERROR PUT : {mse_error_put}")
    print(f"MSE ERROR CALL : {mse_error_call}")

    print(comparison.to_string(index=False))