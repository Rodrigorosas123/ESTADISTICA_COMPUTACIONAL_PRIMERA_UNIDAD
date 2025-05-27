import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Función objetivo
def himmelblau(x, y):
    
    
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

# ---------- ALGORITMO DE LUCIÉRNAGAS ----------
def firefly_algorithm(n=20, max_iter=100, alpha=0.2, beta0=1, gamma=1):
    pos = np.random.uniform(-6, 6, (n, 2))
    light_intensity = np.array([himmelblau(x, y) for x, y in pos])
    history = []

    for t in range(max_iter):
        for i in range(n):
            for j in range(n):
                if light_intensity[j] < light_intensity[i]:
                    r = np.linalg.norm(pos[i] - pos[j])
                    beta = beta0 * np.exp(-gamma * r**2)
                    pos[i] += beta * (pos[j] - pos[i]) + alpha * (np.random.rand(2) - 0.5)
                    light_intensity[i] = himmelblau(*pos[i])
        best_idx = np.argmin(light_intensity)
        history.append((pos[best_idx][0], pos[best_idx][1], light_intensity[best_idx]))

    return pos[best_idx], light_intensity[best_idx], history

# ---------- ALGORITMO PSO ----------
def pso(n=20, max_iter=100, w=0.5, c1=1.5, c2=1.5):
    pos = np.random.uniform(-6, 6, (n, 2))
    vel = np.random.uniform(-1, 1, (n, 2))
    personal_best = pos.copy()
    personal_best_val = np.array([himmelblau(x, y) for x, y in pos])
    global_best = personal_best[np.argmin(personal_best_val)]
    history = []

    for t in range(max_iter):
        for i in range(n):
            r1, r2 = np.random.rand(2)
            vel[i] = (w * vel[i] + c1 * r1 * (personal_best[i] - pos[i]) +
                      c2 * r2 * (global_best - pos[i]))
            pos[i] += vel[i]
            fit = himmelblau(*pos[i])
            if fit < personal_best_val[i]:
                personal_best[i] = pos[i]
                personal_best_val[i] = fit
        global_best = personal_best[np.argmin(personal_best_val)]
        history.append((global_best[0], global_best[1], himmelblau(*global_best)))

    return global_best, himmelblau(*global_best), history

# ---------- ALGORITMO GENÉTICO ----------
def genetic_algorithm(n=20, max_iter=100, mutation_rate=0.1):
    pop = np.random.uniform(-6, 6, (n, 2))
    history = []

    for t in range(max_iter):
        fitness = np.array([himmelblau(x, y) for x, y in pop])
        idx = np.argsort(fitness)
        pop = pop[idx]
        new_pop = pop[:n//2]

        while len(new_pop) < n:
            parents = pop[np.random.choice(n//2, 2, replace=False)]
            child = (parents[0] + parents[1]) / 2
            if np.random.rand() < mutation_rate:
                child += np.random.normal(0, 0.5, 2)
            new_pop = np.vstack((new_pop, child))

        pop = new_pop[:n]
        best_idx = np.argmin([himmelblau(x, y) for x, y in pop])
        best = pop[best_idx]
        history.append((best[0], best[1], himmelblau(*best)))

    return best, himmelblau(*best), history

# ---------- ALGORITMO DE ABEJAS ----------
def bee_algorithm(n=20, max_iter=100, elite_sites=3, elite_bees=5, others=3):
    scouts = np.random.uniform(-6, 6, (n, 2))
    history = []

    for t in range(max_iter):
        fitness = np.array([himmelblau(x, y) for x, y in scouts])
        idx = np.argsort(fitness)
        scouts = scouts[idx]

        new_scouts = []
        for i in range(elite_sites):
            for _ in range(elite_bees):
                patch = scouts[i] + np.random.normal(0, 0.2, 2)
                new_scouts.append(patch)
        for i in range(elite_sites, elite_sites+others):
            for _ in range(2):
                patch = scouts[i] + np.random.normal(0, 0.5, 2)
                new_scouts.append(patch)
        scouts = np.array(new_scouts)
        best_idx = np.argmin([himmelblau(x, y) for x, y in scouts])
        best = scouts[best_idx]
        history.append((best[0], best[1], himmelblau(*best)))

    return best, himmelblau(*best), history

# Ejecutar todos los algoritmos
results = {}
algorithms = {
    "Firefly": firefly_algorithm,
    "PSO": pso,
    "Genetic": genetic_algorithm,
    "Bee": bee_algorithm
}

for name, func in algorithms.items():
    best_pos, best_val, history = func()
    results[name] = {"position": best_pos, "value": best_val, "history": history}

# Graficar evolución de cada algoritmo
plt.figure(figsize=(10, 6))
for name, res in results.items():
    val_history = [v[2] for v in res["history"]]
    plt.plot(val_history, label=name)
plt.title("Evolución del valor óptimo - Función de Himmelblau")
plt.xlabel("Iteraciones")
plt.ylabel("Valor de la función")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Exportar resultados a CSV
df = pd.DataFrame({
    "Algoritmo": [], "X": [], "Y": [], "Valor óptimo": []
})
for name, res in results.items():
    df = pd.concat([df, pd.DataFrame({
        "Algoritmo": [name],
        "X": [res["position"][0]],
        "Y": [res["position"][1]],
        "Valor óptimo": [res["value"]]
    })], ignore_index=True)
df.to_csv("resultados_bioinspirados.csv", index=False)

# Mostrar resumen
print("\nResumen de resultados:")
print(df)
