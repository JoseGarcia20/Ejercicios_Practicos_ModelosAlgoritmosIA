import argparse
import time
from collections import deque

import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

try:  
    import gymnasium as gym 
    GYM_BACKEND = "gymnasium"
except Exception:
    try:
        import gym  
        GYM_BACKEND = "gym"
    except Exception:  
        gym = None
        GYM_BACKEND = None

try:
    import matplotlib.pyplot as plt
except Exception as e:  
    plt = None


def _reset_env(env, seed=None):
    try:
        if seed is not None:
            obs, info = env.reset(seed=seed)
        else:
            obs, info = env.reset()
    except TypeError:
        # Older Gym versions
        if seed is not None:
            try:
                env.seed(seed)
            except Exception:
                pass
        obs = env.reset()
        info = {}
    return obs, info


def _step_env(env, action):
    """Return (obs, reward, done, info) across Gym versions."""
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated) or bool(truncated)
        return obs, reward, done, info, bool(terminated), bool(truncated)
    elif isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        # No separation between terminated/truncated in old API
        return obs, reward, bool(done), info, bool(done), False
    else:  # pragma: no cover
        raise RuntimeError("Respuesta inesperada de env.step")


def epsilon_greedy_action(q_table, state, epsilon, n_actions, rng):
    if rng.random() < epsilon:
        return rng.integers(0, n_actions)
    # Exploit: break ties randomly
    q_row = q_table[state]
    max_q = np.max(q_row)
    candidates = np.flatnonzero(np.isclose(q_row, max_q))
    return int(rng.choice(candidates))

# PARAMETROS DEL APRENDIZAJE
def train_q_learning(
    alpha=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.995,
    episodes=1000,
    max_steps=200,
    seed=123,
    mastery_window=100,
):
    if gym is None:
        raise RuntimeError(
            "No se encontró Gymnasium/Gym. Instala Gymnasium (recomendado):\n"
        )

    env = gym.make("Taxi-v3")

    # Semillas
    rng = np.random.default_rng(seed)
    try:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    except Exception:
        pass

    try:
        n_states = env.observation_space.n
        n_actions = env.action_space.n
    except Exception as e:  # pragma: no cover
        raise RuntimeError("El entorno Taxi-v3 no es discreto.") from e

    q_table = np.zeros((n_states, n_actions), dtype=np.float32)

    epsilon = float(epsilon_start)

    rewards = []
    successes = []  # True si entrega exitosa en el episodio
    q_deltas = []  # Cambio medio absoluto de Q por episodio

    success_count = 0
    success_window = deque(maxlen=mastery_window)
    mastery_episode = None

    t_start = time.perf_counter()
    t_mastery = None

    # Cabecera de la tabla
    print("Episod.\tRecompensa\tExitos\tPasos\%PorConocer\tQΔmed")

    episodes_run = 0
    try:
        for ep in range(1, episodes + 1):
            state, _ = _reset_env(env, seed=seed + ep)
            total_reward = 0.0
            episode_q_delta = 0.0
            steps = 0
            delivered = False

            for t in range(max_steps):
                steps += 1
                action = epsilon_greedy_action(q_table, state, epsilon, n_actions, rng)
                next_state, reward, done, info, terminated, truncated = _step_env(env, action)

                # Q-learning update
                old_q = q_table[state, action]
                td_target = reward + gamma * np.max(q_table[next_state]) * (0.0 if done else 1.0)
                new_q = old_q + alpha * (td_target - old_q)
                q_table[state, action] = new_q
                episode_q_delta += abs(new_q - old_q)

                total_reward += float(reward)
                state = next_state

                if reward == 20 or terminated:
                    delivered = True

                if done:
                    break

            episodes_run = ep
            rewards.append(total_reward)
            successes.append(delivered)
            success_count += 1 if delivered else 0
            success_window.append(1 if delivered else 0)
            mean_q_delta = episode_q_delta / max(1, steps)
            q_deltas.append(mean_q_delta)

            print(
                f"{ep}\t{total_reward:.1f}\t\t{success_count}\t{steps}\t{epsilon:.3f}\t{mean_q_delta:.5f}"
            )

            # Dominio (maestría): todos exitosos en la ventana
            if mastery_episode is None and len(success_window) == mastery_window and sum(success_window) == mastery_window:
                mastery_episode = ep
                t_mastery = time.perf_counter()

            # Decaimiento de exploración
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario. Mostrando resultados...")
    finally:
        env.close()

    t_end = time.perf_counter()

    stats = {
        "episodes_run": episodes_run,
        "rewards": np.array(rewards, dtype=np.float32),
        "successes": np.array(successes, dtype=bool),
        "q_deltas": np.array(q_deltas, dtype=np.float32),
        "q_table": q_table,
        "epsilon_final": epsilon,
        "training_time_sec": t_end - t_start,
        "mastery_episode": mastery_episode,
        "time_to_mastery_sec": (t_mastery - t_start) if t_mastery is not None else None,
        "params": {
            "alpha": alpha,
            "gamma": gamma,
            "epsilon_start": epsilon_start,
            "epsilon_min": epsilon_min,
            "epsilon_decay": epsilon_decay,
            "episodes": episodes,
            "max_steps": max_steps,
            "seed": seed,
            "mastery_window": mastery_window,
        },
    }

    return stats


def plot_results(stats):
    if plt is None:
        print("Matplotlib no está disponible. Instala con: pip install matplotlib")
        return

    rewards = stats["rewards"]
    q_deltas = stats["q_deltas"]
    successes = stats["successes"]
    episodes = np.arange(1, len(rewards) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    # Recompensas por episodio
    ax = axes[0]
    ax.plot(episodes, rewards, color="tab:blue", linewidth=1, label="Recompensa por episodio")
    # Media móvil para suavizar
    if len(rewards) >= 20:
        k = 20
        kernel = np.ones(k) / k
        smoothed = np.convolve(rewards, kernel, mode="valid")
        ax.plot(np.arange(k, len(rewards) + 1), smoothed, color="tab:orange", label=f"Media móvil ({k})")
    ax.set_title("Recompensas vs Episodios (Taxi-v3)")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Recompensa")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Estabilidad de Q (cambio medio por paso)
    ax2 = axes[1]
    ax2.plot(episodes, q_deltas, color="tab:green", linewidth=1, label="|ΔQ| medio por episodio")
    ax2.set_title("Estabilidad de los valores Q (convergencia)")
    ax2.set_xlabel("Episodio")
    ax2.set_ylabel("|ΔQ| medio")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    # Mostrar tasa de éxito acumulada como texto
    success_rate = 100.0 * successes.mean() if len(successes) else 0.0
    title = f"Episodios: {len(rewards)} | Éxitos: {successes.sum()} ({success_rate:.1f}%)"
    if stats.get("mastery_episode") is not None:
        title += f" | Maestría en ep {stats['mastery_episode']}"
    fig.suptitle(title)

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Q-Learning en Taxi-v3 (Gym) con métricas de recompensa, estabilidad de Q y tiempo de aprendizaje. "
            "Use Ctrl+C para detener y ver gráficas."
        )
    )
    parser.add_argument("--episodes", type=int, default=1000, help="Número de episodios a entrenar")
    parser.add_argument("--max-steps", type=int, default=200, help="Máximo de pasos por episodio")
    parser.add_argument("--alpha", type=float, default=0.1, help="Tasa de aprendizaje (α)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Factor de descuento (γ)")
    parser.add_argument("--epsilon-start", type=float, default=0.2, help="Exploración inicial (ε)")
    parser.add_argument("--epsilon-min", type=float, default=0.05, help="Exploración mínima")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Decaimiento por episodio de ε")
    parser.add_argument("--seed", type=int, default=123, help="Semilla aleatoria")
    parser.add_argument(
        "--mastery-window",
        type=int,
        default=100,
        help="Tamaño de ventana para detectar dominio (todos exitosos en la ventana)",
    )

    args = parser.parse_args()

    stats = train_q_learning(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        mastery_window=args.mastery_window,
    )

    # Resumen final
    print("\nResumen de entrenamiento:")
    print(f"Episodios ejecutados: {stats['episodes_run']}")
    print(f"Éxitos totales: {int(stats['successes'].sum())} / {stats['episodes_run']}")
    if stats.get("mastery_episode") is not None:
        print(
            f"Dominio alcanzado en episodio {stats['mastery_episode']} en "
            f"{stats['time_to_mastery_sec']:.2f}s"
        )
    print(f"Tiempo total: {stats['training_time_sec']:.2f}s")
    print(f"Epsilon final: {stats['epsilon_final']:.3f}")

    # Graficar al finalizar o si se interrumpió
    plot_results(stats)


if __name__ == "__main__":
    main()
