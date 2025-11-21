#%%
import numpy as np

#%%
def simulate_gbm_random_walk(S0, r, sigma, T, n_steps, rng=None):
    """
    Simule UN chemin de prix sous Black-Scholes par marche aléatoire discrète (GBM).

    S_{t+dt} = S_t * exp( (r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z_t )

    Args:
        S0      : prix initial
        r       : taux sans risque (continu)
        sigma   : volatilité
        T       : maturité (années)
        n_steps : nombre de pas de temps
        rng     : générateur numpy (optionnel)

    Returns:
        np.ndarray de taille (n_steps+1,) contenant le chemin S_0, S_dt, ..., S_T
    """
    if rng is None:
        rng = np.random.default_rng()

    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    vol   = sigma * np.sqrt(dt)

    # tableau pour le chemin
    S = np.empty(n_steps + 1)
    S[0] = S0

    for t in range(1, n_steps + 1):
        Z = rng.standard_normal()              # Z ~ N(0,1)
        S[t] = S[t-1] * np.exp(drift + vol*Z)  # marche discrète multiplicative

    return S

#%%
def monte_carlo_option_from_random_walk(
    S0, K, r, sigma, T,
    n_steps, n_sim,
    payoff_type="put"
):
    """
    Monte Carlo qui utilise la marche aléatoire (random walk) pour pricer une option européenne.

    Args:
        S0, K, r, sigma, T : paramètres Black-Scholes
        n_steps : nombre de pas dans la marche discrète
        n_sim   : nombre de trajectoires Monte Carlo
        payoff_type : "put" ou "call"

    Returns:
        float : prix estimé de l'option
    """
    rng = np.random.default_rng()
    payoffs = np.empty(n_sim)

    for i in range(n_sim):
        path = simulate_gbm_random_walk(S0, r, sigma, T, n_steps, rng=rng)
        ST = path[-1]  # prix à maturité

        if payoff_type == "put":
            payoffs[i] = max(K - ST, 0.0)
        elif payoff_type == "call":
            payoffs[i] = max(ST - K, 0.0)
        else:
            raise ValueError("payoff_type doit être 'put' ou 'call'.")

    price = np.exp(-r * T) * payoffs.mean()
    return float(price)

# %%
# Exemple d'utilisation :
if __name__ == "__main__":
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
    n_steps = 252   # ~ 252 jours de bourse
    n_sim   = 50_000

    put_price  = monte_carlo_option_from_random_walk(S0, K, r, sigma, T, n_steps, n_sim, payoff_type="put")
    call_price = monte_carlo_option_from_random_walk(S0, K, r, sigma, T, n_steps, n_sim, payoff_type="call")

    print("Put (MC + random walk)  ≈", put_price)
    print("Call (MC + random walk) ≈", call_price)
# %%
