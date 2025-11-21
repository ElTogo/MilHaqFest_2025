import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from math import sqrt, exp

def heston_classical_mc(S0, K, T, r, v0, theta, kappa, eta, rho, 
                        T_days=182, Sbar=130.0, Tbar=50, Nsim=1000):
    """
    Simulation Monte Carlo classique du modèle de Heston avec barrière parisienne
    """
    h = 1/365  # pas de temps journalier
    dim = T_days + 1
    
    rng = np.random.default_rng(42)
    
    X = np.full((Nsim,), np.log(S0))
    v = np.full((Nsim,), v0)
    
    # Compteur d'excursions consécutives au-dessus de la barrière
    excursion = np.zeros(Nsim, dtype=np.int32)
    knocked = np.zeros(Nsim, dtype=bool)
    
    for j in range(1, dim):
        Zs = rng.standard_normal(Nsim)
        Zv_perp = rng.standard_normal(Nsim)
        Zv = rho * Zs + np.sqrt(1.0 - rho**2) * Zv_perp
        
        v_clipped = np.maximum(v, 0.0)
        
        X = X + (r - 0.5 * v_clipped) * h + np.sqrt(v_clipped * h) * Zs
        v = v + kappa * (theta - v) * h + eta * np.sqrt(v_clipped * h) * Zv
        v = np.maximum(v, 0.0)
        
        S = np.exp(X)
        above = S >= Sbar
        excursion = np.where(above, excursion + 1, 0)
        knocked |= (excursion > Tbar)
    
    ST = np.exp(X)
    disc = np.exp(-r * T)
    
    # Prix européen simple
    euroCall = disc * np.mean(np.maximum(ST - K, 0.0))
    euroPut = disc * np.mean(np.maximum(K - ST, 0.0))
    
    # Prix Parisian Up-and-Out
    payoff_parisian_call = np.where(knocked, 0.0, np.maximum(ST - K, 0.0))
    payoff_parisian_put = np.where(knocked, 0.0, np.maximum(K - ST, 0.0))
    parisianCall_UaO = disc * payoff_parisian_call.mean()
    parisianPut_UaO = disc * payoff_parisian_put.mean()
    
    # Statistiques
    nb_knocked = int(np.count_nonzero(knocked))
    ratio_knocked = nb_knocked / len(knocked)
    std_error_call = (disc * payoff_parisian_call).std(ddof=1) / np.sqrt(Nsim)
    std_error_put = (disc * payoff_parisian_put).std(ddof=1) / np.sqrt(Nsim)
    
    return {
        'euroCall': euroCall,
        'euroPut': euroPut,
        'parisianCall_UaO': parisianCall_UaO,
        'parisianPut_UaO': parisianPut_UaO,
        'nb_knocked': nb_knocked,
        'ratio_knocked': ratio_knocked,
        'std_error_call': std_error_call,
        'std_error_put': std_error_put,
        'final_prices': ST,
        'knocked': knocked
    }

def heston_quantum_mc(S0, K, T, r, v0, theta, kappa, eta, rho,
                      T_days=182, Sbar=130.0, Tbar=50, 
                      n_qubits=6, shots=10000, n_classical=100):
    """
    Simulation hybride Heston: évolution classique + sampling quantique des trajectoires finales
    
    Args:
        n_qubits: nombre de qubits pour encoder les trajectoires
        shots: nombre de mesures quantiques
        n_classical: nombre de trajectoires classiques à générer (base du sampling)
    """
    h = 1/365
    dim = T_days + 1
    
    rng = np.random.default_rng(42)
    
    # 1) Générer n_classical trajectoires classiques complètes
    print(f"  Génération de {n_classical} trajectoires classiques...")
    X_paths = np.full((n_classical,), np.log(S0))
    v_paths = np.full((n_classical,), v0)
    excursion_paths = np.zeros(n_classical, dtype=np.int32)
    knocked_paths = np.zeros(n_classical, dtype=bool)
    
    for j in range(1, dim):
        Zs = rng.standard_normal(n_classical)
        Zv_perp = rng.standard_normal(n_classical)
        Zv = rho * Zs + np.sqrt(1.0 - rho**2) * Zv_perp
        
        v_clipped = np.maximum(v_paths, 0.0)
        
        X_paths = X_paths + (r - 0.5 * v_clipped) * h + np.sqrt(v_clipped * h) * Zs
        v_paths = v_paths + kappa * (theta - v_paths) * h + eta * np.sqrt(v_clipped * h) * Zv
        v_paths = np.maximum(v_paths, 0.0)
        
        S_temp = np.exp(X_paths)
        above = S_temp >= Sbar
        excursion_paths = np.where(above, excursion_paths + 1, 0)
        knocked_paths |= (excursion_paths > Tbar)
    
    ST_classical = np.exp(X_paths)
    
    # 2) Calculer les payoffs pour chaque trajectoire
    payoff_euro_call = np.maximum(ST_classical - K, 0.0)
    payoff_euro_put = np.maximum(K - ST_classical, 0.0)
    payoff_parisian_call = np.where(knocked_paths, 0.0, np.maximum(ST_classical - K, 0.0))
    payoff_parisian_put = np.where(knocked_paths, 0.0, np.maximum(K - ST_classical, 0.0))
    
    # 3) Créer une distribution de probabilité basée sur les payoffs (on utilise call)
    weights = payoff_parisian_call + 1e-10  # éviter division par zéro
    probs = weights / weights.sum()
    
    # 4) Encoder dans un circuit quantique
    print(f"  Encodage quantique avec {n_qubits} qubits...")
    n_states = 2**n_qubits
    
    if n_classical < n_states:
        # Padding avec des zéros
        probs_padded = np.zeros(n_states)
        probs_padded[:n_classical] = probs
        payoff_padded_call = np.zeros(n_states)
        payoff_padded_call[:n_classical] = payoff_parisian_call
        payoff_padded_put = np.zeros(n_states)
        payoff_padded_put[:n_classical] = payoff_parisian_put
        payoff_euro_call_padded = np.zeros(n_states)
        payoff_euro_call_padded[:n_classical] = payoff_euro_call
        payoff_euro_put_padded = np.zeros(n_states)
        payoff_euro_put_padded[:n_classical] = payoff_euro_put
    else:
        # Échantillonner ou regrouper
        indices = np.linspace(0, n_classical-1, n_states).astype(int)
        probs_padded = probs[indices]
        probs_padded = probs_padded / probs_padded.sum()
        payoff_padded_call = payoff_parisian_call[indices]
        payoff_padded_put = payoff_parisian_put[indices]
        payoff_euro_call_padded = payoff_euro_call[indices]
        payoff_euro_put_padded = payoff_euro_put[indices]
    
    # Amplitudes = sqrt(probabilités)
    amplitudes = np.sqrt(probs_padded)
    amplitudes = amplitudes / np.linalg.norm(amplitudes)
    
    # 5) Circuit quantique
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.initialize(amplitudes, list(range(n_qubits)))
    qc.measure(list(range(n_qubits)), list(range(n_qubits)))
    
    # 6) Simulation
    print(f"  Simulation quantique avec {shots} shots...")
    simulator = AerSimulator()
    transpiled_qc = transpile(qc, simulator)
    job = simulator.run(transpiled_qc, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    # 7) Calcul du prix à partir des mesures quantiques
    disc = np.exp(-r * T)
    total_shots = sum(counts.values())
    exp_payoff_euro_call = 0.0
    exp_payoff_euro_put = 0.0
    exp_payoff_parisian_call = 0.0
    exp_payoff_parisian_put = 0.0
    
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        if idx < len(payoff_padded_call):
            exp_payoff_parisian_call += payoff_padded_call[idx] * count
            exp_payoff_parisian_put += payoff_padded_put[idx] * count
            exp_payoff_euro_call += payoff_euro_call_padded[idx] * count
            exp_payoff_euro_put += payoff_euro_put_padded[idx] * count
    
    exp_payoff_euro_call /= total_shots
    exp_payoff_euro_put /= total_shots
    exp_payoff_parisian_call /= total_shots
    exp_payoff_parisian_put /= total_shots
    
    euroCall_quantum = disc * exp_payoff_euro_call
    euroPut_quantum = disc * exp_payoff_euro_put
    parisianCall_quantum = disc * exp_payoff_parisian_call
    parisianPut_quantum = disc * exp_payoff_parisian_put
    
    # Statistiques
    nb_knocked = int(np.count_nonzero(knocked_paths))
    ratio_knocked = nb_knocked / len(knocked_paths)
    
    # Estimation de l'erreur (approximative)
    payoff_samples_call = []
    payoff_samples_put = []
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        if idx < len(payoff_padded_call):
            payoff_samples_call.extend([payoff_padded_call[idx]] * count)
            payoff_samples_put.extend([payoff_padded_put[idx]] * count)
    
    std_error_call = np.std(payoff_samples_call, ddof=1) / np.sqrt(len(payoff_samples_call)) * disc
    std_error_put = np.std(payoff_samples_put, ddof=1) / np.sqrt(len(payoff_samples_put)) * disc
    
    return {
        'euroCall': euroCall_quantum,
        'euroPut': euroPut_quantum,
        'parisianCall_UaO': parisianCall_quantum,
        'parisianPut_UaO': parisianPut_quantum,
        'nb_knocked': nb_knocked,
        'ratio_knocked': ratio_knocked,
        'std_error_call': std_error_call,
        'std_error_put': std_error_put,
        'circuit_depth': qc.depth(),
        'n_classical_paths': n_classical
    }

def process_heston_csv(csv_file, output_file, method="classical"):
    """
    Traite un CSV avec les paramètres Heston et calcule les prix
    
    Args:
        csv_file: fichier CSV avec colonnes r, K, S0, Sigma, T
        output_file: fichier de sortie
        method: "classical" ou "quantum"
    """
    # Lire le CSV avec le bon séparateur
    df = pd.read_csv(csv_file, sep=',')
    
    # Remplacer virgules par points si nécessaire
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
            except:
                pass
    
    results = []
    
    print(f"{'='*80}")
    print(f"CALCUL DES PRIX HESTON - MÉTHODE: {method.upper()}")
    print(f"{'='*80}\n")
    
    for idx, row in df.iterrows():
        print(f"\nLigne {idx+1}:")
        
        # Paramètres de base depuis le CSV
        r = row['r']
        K = row['K']
        S0 = row['S0']
        sigma = row['Sigma']
        T = row['T']
        
        print(f"  S0={S0:.3f}, K={K:.3f}, r={r:.4f}, σ={sigma:.4f}, T={T:.1f}")
        
        # Paramètres Heston (depuis CSV ou par défaut)
        v0 = row.get('v0', sigma**2)
        theta = row.get('theta', 0.1)
        kappa = row.get('kappa', 2.0)
        eta = row.get('eta', 0.40)
        rho = row.get('rho', -0.5)
        
        # Paramètres de barrière
        T_days = int(T * 365)
        Sbar = row.get('Sbar', S0 * 1.3)
        Tbar = row.get('Tbar', 50)
        
        if method == "classical":
            result = heston_classical_mc(
                S0, K, T, r, v0, theta, kappa, eta, rho,
                T_days=T_days, Sbar=Sbar, Tbar=Tbar, Nsim=1000
            )
        else:  # quantum
            result = heston_quantum_mc(
                S0, K, T, r, v0, theta, kappa, eta, rho,
                T_days=T_days, Sbar=Sbar, Tbar=Tbar,
                n_qubits=6, shots=10000, n_classical=100
            )
        
        print(f"  Call européen: {result['euroCall']:.6f}")
        print(f"  Put européen: {result['euroPut']:.6f}")
        print(f"  Parisian Call UaO: {result['parisianCall_UaO']:.6f}")
        print(f"  Parisian Put UaO: {result['parisianPut_UaO']:.6f}")
        print(f"  Knocked: {result['nb_knocked']} ({result['ratio_knocked']:.2%})")
        
        results.append({
            'r': r,
            'K': K,
            'S0': S0,
            'Sigma': sigma,
            'T': T,
            'Euro_Call': result['euroCall'],
            'Euro_Put': result['euroPut'],
            'Parisian_Call_UaO': result['parisianCall_UaO'],
            'Parisian_Put_UaO': result['parisianPut_UaO'],
            'Knocked_Ratio': result['ratio_knocked'],
            'Std_Error_Call': result['std_error_call'],
            'Std_Error_Put': result['std_error_put']
        })
    
    # Sauvegarder avec virgules comme séparateur décimal pour Excel
    df_results = pd.DataFrame(results)
    
    # Formater les colonnes numériques
    numeric_cols = ['Euro_Call', 'Euro_Put', 'Parisian_Call_UaO', 'Parisian_Put_UaO', 
                    'Knocked_Ratio', 'Std_Error_Call', 'Std_Error_Put']
    
    df_output = df_results.copy()
    for col in numeric_cols:
        if col in df_output.columns:
            df_output[col] = df_output[col].apply(lambda x: f"{x:.6f}".replace('.', ','))
    
    df_output.to_csv(output_file, sep=',', index=False)
    
    print(f"\n{'='*80}")
    print(f"Résultats sauvegardés dans: {output_file}")
    print(f"{'='*80}")
    
    return df_results

# Exemple d'utilisation
if __name__ == "__main__":
    import sys
    
    # Vérifier si un fichier CSV est fourni en argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        method = sys.argv[2] if len(sys.argv) > 2 else "classical"
        
        print(f"\n### TRAITEMENT DU FICHIER: {input_file} ###\n")
        
        if method == "classical":
            output_file = "heston_results_classical.csv"
        else:
            output_file = "heston_results_quantum.csv"
        
        df_results = process_heston_csv(input_file, output_file, method=method)
        
    else:
        # Test sur un exemple simple
        print("\n### TEST UNIQUE ###\n")
        
        params = {
            'S0': 100, 'K': 100, 'T': 1, 'r': 0.1,
            'v0': 0.0625, 'theta': 0.1, 'kappa': 2, 
            'eta': 0.40, 'rho': -0.5,
            'T_days': 182, 'Sbar': 130, 'Tbar': 50
        }
        
        print("1. Simulation Classique:")
        result_classical = heston_classical_mc(**params, Nsim=1000)
        print(f"   Call européen: {result_classical['euroCall']:.6f}")
        print(f"   Put européen: {result_classical['euroPut']:.6f}")
        print(f"   Parisian Call UaO: {result_classical['parisianCall_UaO']:.6f}")
        print(f"   Parisian Put UaO: {result_classical['parisianPut_UaO']:.6f}")
        print(f"   Ratio knocked: {result_classical['ratio_knocked']:.2%}")
        
        print("\n2. Simulation Quantique:")
        result_quantum = heston_quantum_mc(
            **{k: params[k] for k in ['S0', 'K', 'T', 'r', 'v0', 'theta', 'kappa', 'eta', 'rho']},
            T_days=params['T_days'], Sbar=params['Sbar'], Tbar=params['Tbar'],
            n_qubits=6, shots=10000, n_classical=100
        )
        print(f"   Call européen: {result_quantum['euroCall']:.6f}")
        print(f"   Put européen: {result_quantum['euroPut']:.6f}")
        print(f"   Parisian Call UaO: {result_quantum['parisianCall_UaO']:.6f}")
        print(f"   Parisian Put UaO: {result_quantum['parisianPut_UaO']:.6f}")
        print(f"   Profondeur circuit: {result_quantum['circuit_depth']}")
        
        # Demander si l'utilisateur veut traiter un CSV
        print("\n" + "="*80)
        process_csv = input("Voulez-vous traiter un fichier CSV? (y/n): ").lower() == 'y'
        
        if process_csv:
            csv_path = input("Chemin du fichier CSV: ").strip('"')
            method_choice = input("Méthode (classical/quantum) [classical]: ").lower() or "classical"
            
            output_name = f"heston_results_{method_choice}.csv"
            df_results = process_heston_csv(csv_path, output_name, method=method_choice)