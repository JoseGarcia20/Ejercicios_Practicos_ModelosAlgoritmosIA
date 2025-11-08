import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.neighbors import KDTree

# Cargar datos
housing = fetch_california_housing(as_frame=True)
df = housing.frame.copy()

# Orden caracteristicas tabla
FEAT_ORDER = [
    'MedInc',     # ingreso medio
    'HouseAge',   # antigüedad media
    'AveRooms',   # habitaciones promedio por hogar
    'AveBedrms',  # dormitorios promedio por hogar
    'Population', # población del bloque
    'AveOccup',   # ocupantes promedio por hogar
    'Latitude',   # latitud
    'Longitude'   # longitud (negativa en CA)
]

X = df[FEAT_ORDER].to_numpy()
y = df['MedHouseVal'].to_numpy()
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Ecuación normal (usamos pseudoinversa por
theta_best = np.linalg.pinv(X_b.T @ X_b) @ (X_b.T @ y)

# KDTree para búsqueda de vecinos por (lat, lon)
tree = KDTree(df[['Latitude', 'Longitude']].to_numpy())

# Función de predicción por coordenadas con aleatoriedad - Monte Carlo
def predict_price_by_coords(
    lat: float,
    lon: float,
    *,
    k: int = 60,                 # # de vecinos geográficos
    n_samples: int = 1500,       # simulaciones Monte Carlo
    noise_scale: float = 0.07,   # 7% de ruido relativo
    clip_target: bool = True,    # recortar a [0,5] como el dataset
    random_state: int | None = 123
):

    rng = np.random.default_rng(random_state)

    # Vecinos más cercanos por coordenadas
    dists, idxs = tree.query([[lat, lon]], k=k, return_distance=True)
    idxs = idxs[0]; dists = dists[0]

    neigh = df.iloc[idxs][FEAT_ORDER].to_numpy()

    # Ponderación por distancia (inverso; +epsilon)
    w = 1.0 / (dists + 1e-6)
    w = w / w.sum()

    # Features base (promedio ponderado)
    base = (neigh * w[:, None]).sum(axis=0)

    # Covarianza local y suavizado
    if len(neigh) > 3:
        cov = np.cov(neigh, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6  # estabilizador
    else:
        # Diagonal con varianzas simples
        cov = np.diag(np.var(neigh, axis=0, ddof=1))

    cov_scaled = cov * (noise_scale ** 2)

    # Simulaciones de features alrededor del "base"
    sims = rng.multivariate_normal(mean=base, cov=cov_scaled, size=n_samples)

    # Recorte por-feature (evita imposibles pero respeta lat/lon)
    # Rangos razonables para California Housing
    lb = np.array([0,    0,    0,    0,    0,    0,   32.0,  -125.0])
    ub = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 42.5, -114.0])
    sims = np.clip(sims, lb, ub)

    # Armar matriz con bias y predecir con θ
    sims_b = np.c_[np.ones((n_samples, 1)), sims]
    y_sims = sims_b @ theta_best  # centenas de miles

    if clip_target:
        y_sims = np.clip(y_sims, 0, 5)  # dataset capado a [0,5]

    # Estadísticos
    y_mean = float(np.mean(y_sims))
    y_std  = float(np.std(y_sims, ddof=1))
    ci_low, ci_high = np.percentile(y_sims, [2.5, 97.5])
    y_random = float(rng.choice(y_sims))

    base_features = dict(zip(FEAT_ORDER, base))
    return {
        "y_mean": y_mean,
        "y_std": y_std,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "y_random": y_random,
        "base_features": base_features,
        "neighbors_used": int(k),
    }

# Función ayudante: pretty print en USD
def print_prediction_result(lat, lon, res):
    print(f"Coordenadas consultadas: (lat={lat}, lon={lon})")
    print("\nFeatures base estimadas (promedio ponderado de vecinos):")
    for kf, vf in res["base_features"].items():
        print(f"  {kf:10s}: {vf:.4f}")
    print("\nPredicción (cientos de miles USD):")
    print(f"  Media     : {res['y_mean']:.3f}")
    print(f"  Desv. Est.: {res['y_std']:.3f}")
    print(f"  IC 95%    : [{res['ci_low']:.3f}, {res['ci_high']:.3f}]")
    print(f"  Aleatoria : {res['y_random']:.3f}")

    print("\nEn USD aproximados:")
    print(f"  Media     : ${res['y_mean']*100000:,.0f}")
    print(f"  IC 95%    : [${res['ci_low']*100000:,.0f}, ${res['ci_high']*100000:,.0f}]")
    print(f"  Aleatoria : ${res['y_random']*100000:,.0f}")

# Ejemplo de uso
if __name__ == "__main__":

    # Ejemplos dentro de California:
    ejemplos = [
        ("Los Ángeles (centro aprox.)", 34.05, -118.25)
        #("San José (Silicon Valley)",   37.34, -121.89),
        #("Beverly Hills",                   34.0736, -118.4004),
    ]

    for nombre, lat, lon in ejemplos:
        print("\n" + "="*70)
        print(f"Ejemplo: {nombre}")
        res = predict_price_by_coords(
            lat, lon,
            k=60,               # vecinos
            n_samples=1500,     # simulaciones
            noise_scale=0.07,   # ruido relativo
            clip_target=True,   # respeta el cap [0,5]
            random_state=123
        )
        print_prediction_result(lat, lon, res)
