"""
Implementación de los tres algoritmos de Tarea 2 para el Problema del Agente Viajero (TSP):
1. Búsqueda Local (2-opt Hill Climbing)
2. Recocido Simulado (SA) + 2-opt
3. Aproximación MST (Minimum Spanning Tree) + 2-opt
"""
import math
import random


def dist(p: tuple[float, float], q: tuple[float, float]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def tour_cost(tour: list[int], points: list[tuple[float, float]]) -> float:
    n = len(tour)
    if n <= 1:
        return 0.0
    return sum(dist(points[tour[i]], points[tour[(i + 1) % n]]) for i in range(n))


def nearest_neighbor_tour(points: list[tuple[float, float]]) -> list[int]:
    n = len(points)
    unvisited = set(range(n))
    current = 0
    unvisited.remove(current)
    tour = [current]
    while unvisited:
        nxt = min(unvisited, key=lambda j: dist(points[current], points[j]))
        tour.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return tour


def two_opt_local_search(tour: list[int], points: list[tuple[float, float]]) -> list[int]:
    """Búsqueda local 2-opt (hill climbing) hasta óptimo local."""
    n = len(tour)
    if n < 4:
        return tour[:]
    tour = tour[:]
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = (dist(points[a], points[c]) + dist(points[b], points[d])) - (
                    dist(points[a], points[b]) + dist(points[c], points[d])
                )
                if delta < -1e-12:
                    tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
                    improved = True
    return tour


def solve_2opt_hill_climbing(points: list[tuple[float, float]]) -> tuple[list[int], float]:
    """Algoritmo 1: Construcción Nearest Neighbor + 2-opt Hill Climbing."""
    initial = nearest_neighbor_tour(points)
    best = two_opt_local_search(initial, points)
    return best, tour_cost(best, points)


def solve_sa_2opt(
    points: list[tuple[float, float]],
    seed: int = 42,
    t0: float = 15.0,
    alpha: float = 0.995,
    iters: int | None = None,
) -> tuple[list[int], float]:
    """
    Algoritmo 2: Recocido Simulado (Simulated Annealing) con vecindario 2-opt + pulido 2-opt.
    Parámetros documentados:
    - Temperatura inicial (t0): 15.0
    - Esquema de enfriamiento geométrico: T_{k+1} = alpha * T_k con alpha = 0.995
    - Iteraciones por instancia: max(500, 150 * n)
    - Refinamiento final: búsqueda local 2-opt sobre la mejor solución hallada
    """
    rng = random.Random(seed)
    n = len(points)
    if iters is None:
        iters = max(500, 150 * n)

    tour = nearest_neighbor_tour(points)
    cost = tour_cost(tour, points)
    best_tour, best_cost = tour[:], cost
    t = t0

    for _ in range(iters):
        if n < 4:
            break
        i = rng.randint(0, n - 2)
        j = rng.randint(i + 1, n - 1)
        if i == 0 and j == n - 1:
            continue
        a, b = tour[i], tour[(i + 1) % n]
        c, d = tour[j], tour[(j + 1) % n]
        delta = (dist(points[a], points[c]) + dist(points[b], points[d])) - (
            dist(points[a], points[b]) + dist(points[c], points[d])
        )
        if delta < 0 or rng.random() < math.exp(-delta / max(t, 1e-9)):
            tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
            cost += delta
            if cost < best_cost:
                best_tour, best_cost = tour[:], cost
        t *= alpha

    polished = two_opt_local_search(best_tour, points)
    return polished, tour_cost(polished, points)


def solve_mst_2opt(points: list[tuple[float, float]]) -> tuple[list[int], float]:
    """Algoritmo 3: Árbol de Expansión Mínima (Prim) + DFS preorden + refinamiento 2-opt."""
    n = len(points)
    in_tree = [False] * n
    min_edge = [math.inf] * n
    parent = [-1] * n
    min_edge[0] = 0.0
    adj = [[] for _ in range(n)]

    for _ in range(n):
        u = min((v for v in range(n) if not in_tree[v]), key=lambda v: min_edge[v])
        in_tree[u] = True
        if parent[u] != -1:
            adj[u].append(parent[u])
            adj[parent[u]].append(u)
        for v in range(n):
            if not in_tree[v]:
                d = dist(points[u], points[v])
                if d < min_edge[v]:
                    min_edge[v] = d
                    parent[v] = u

    preorder, visited = [], [False] * n

    def dfs(u: int):
        visited[u] = True
        preorder.append(u)
        for v in adj[u]:
            if not visited[v]:
                dfs(v)

    dfs(0)
    polished = two_opt_local_search(preorder, points)
    return polished, tour_cost(polished, points)
