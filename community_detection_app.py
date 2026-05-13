#!/usr/bin/env python3
"""
Community Detection using Genetic Algorithms
Lab 11 - Algoritmi Evolutivi | 

Referinta: Pizzuti, Clara. "Evolutionary computation for community detection in networks: a review."
           IEEE Transactions on Evolutionary Computation 22.3 (2017): 464-483.

Utilizare:
    python community_detection_app.py                          # demo cu Karate Club
    python community_detection_app.py --graph retea.gml        # un singur fisier GML
    python community_detection_app.py --graphs_dir ./networks  # director cu fisiere GML
    python community_detection_app.py --graph retea.gml --fitness conductance --pop_size 150 --generations 300
"""

import os
import sys
import time
import random
import argparse
import warnings
from collections import defaultdict

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# SECTION 1: REPREZENTARE CROMOZOM (Locus-based Adjacency)
# ─────────────────────────────────────────────────────────────

def build_neighbor_index(graph, nodes, node_to_idx):
    """Construieste indexul vecinilor pentru fiecare nod (ca indici)."""
    neighbors = {}
    for node in nodes:
        i = node_to_idx[node]
        neighbors[i] = [node_to_idx[nb] for nb in graph.neighbors(node)]
    return neighbors


def random_chromosome(n, neighbors):
    """
    Codificare locus-based adjacency (Pizzuti 2017).
    Gene i = un vecin aleator al nodului i.
    Daca nodul nu are vecini, gene[i] = i (self-loop).
    """
    chrom = list(range(n))
    for i in range(n):
        if neighbors[i]:
            chrom[i] = random.choice(neighbors[i])
    return chrom


def decode_chromosome(chromosome):
    """
    Decodifica cromozomul in etichete de comunitate.
    Se construieste un graf unde nodul i -> chromosome[i],
    iar componentele conexe = comunitatile.
    Foloseste Union-Find pentru eficienta O(n * alpha(n)).
    """
    n = len(chromosome)
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1

    for i, j in enumerate(chromosome):
        union(i, j)

    root_to_id = {}
    communities = [0] * n
    for i in range(n):
        root = find(i)
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id)
        communities[i] = root_to_id[root]

    return communities


def chromosome_to_partition(chromosome, nodes):
    """Converteste cromozomul in lista de multimi (partitie)."""
    comm_labels = decode_chromosome(chromosome)
    partition = defaultdict(set)
    for node_idx, comm_id in enumerate(comm_labels):
        partition[comm_id].add(nodes[node_idx])
    return list(partition.values())


# ─────────────────────────────────────────────────────────────
# SECTION 2: FUNCTII DE FITNESS
# ─────────────────────────────────────────────────────────────

def fitness_modularity(chromosome, graph, nodes, **kwargs):
    """
    Fitness 1: Modularitate Newman-Girvan (Q).
    Q = sum_c [ L_c/m - (d_c / 2m)^2 ]
    Valori bune: Q > 0.3
    """
    partition = chromosome_to_partition(chromosome, nodes)
    try:
        return nx.community.modularity(graph, partition)
    except Exception:
        return -1.0


def fitness_conductance(chromosome, graph, nodes, **kwargs):
    """
    Fitness 2: Conductanta medie negativa.
    Conductance(S) = cut(S, V\\S) / min(vol(S), vol(V\\S))
    Conductanta mica = comunitati bune => maximizam -conductanta_medie.
    """
    partition = chromosome_to_partition(chromosome, nodes)
    if len(partition) <= 1:
        return -1.0

    node_set = set(nodes)
    conductances = []
    for community in partition:
        if not community or len(community) >= len(nodes):
            continue
        complement = node_set - community
        cut = nx.cut_size(graph, community, complement)
        vol_s = sum(d for _, d in graph.degree(community))
        vol_t = sum(d for _, d in graph.degree(complement))
        denom = min(vol_s, vol_t)
        if denom > 0:
            conductances.append(cut / denom)

    if not conductances:
        return -1.0
    return -float(np.mean(conductances))  # negate pentru maximizare


def fitness_coverage_performance(chromosome, graph, nodes, **kwargs):
    """
    Fitness 3: Media dintre Coverage si Performance.
    Coverage  = muchii intra-comunitate / total muchii
    Performance = perechi corect clasificate / total perechi
    Ambele in [0,1], mai mare = mai bine.
    """
    partition = chromosome_to_partition(chromosome, nodes)
    try:
        coverage, performance = nx.community.partition_quality(graph, partition)
        return float((coverage + performance) / 2.0)
    except Exception:
        return -1.0


FITNESS_FUNCTIONS = {
    'modularity':           fitness_modularity,
    'conductance':          fitness_conductance,
    'coverage_performance': fitness_coverage_performance,
}

FITNESS_DESCRIPTIONS = {
    'modularity':           'Modularitate Q (Newman-Girvan)',
    'conductance':          'Conductanta medie negativa',
    'coverage_performance': 'Coverage + Performance (medie)',
}


# ─────────────────────────────────────────────────────────────
# SECTION 3: OPERATORI GENETICI
# ─────────────────────────────────────────────────────────────

def tournament_selection(population, fitnesses, k=3):
    """Selectie prin turneu de dimensiune k."""
    candidates = random.sample(range(len(population)), min(k, len(population)))
    best = max(candidates, key=lambda x: fitnesses[x])
    return population[best][:]


def one_point_crossover(p1, p2, rate=0.8):
    """Incrucisare intr-un punct."""
    if random.random() > rate:
        return p1[:], p2[:]
    pt = random.randint(1, len(p1) - 1)
    return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]


def uniform_crossover(p1, p2, rate=0.8):
    """Incrucisare uniforma (fiecare gena cu prob 0.5)."""
    if random.random() > rate:
        return p1[:], p2[:]
    c1, c2 = p1[:], p2[:]
    for i in range(len(p1)):
        if random.random() < 0.5:
            c1[i], c2[i] = c2[i], c1[i]
    return c1, c2


def mutate(chromosome, neighbors, rate=0.05):
    """Mutatie: inlocuire gena cu vecin aleator."""
    for i in range(len(chromosome)):
        if random.random() < rate and neighbors[i]:
            chromosome[i] = random.choice(neighbors[i])
    return chromosome


def repair(chromosome, neighbors):
    """
    Reparare cromozom: asigura ca fiecare gena i
    contine un vecin valid al nodului i.
    """
    neighbor_sets = {i: set(nb_list) for i, nb_list in neighbors.items()}
    for i in range(len(chromosome)):
        if chromosome[i] != i and chromosome[i] not in neighbor_sets[i]:
            chromosome[i] = random.choice(neighbors[i]) if neighbors[i] else i
    return chromosome


CROSSOVER_FUNCTIONS = {
    'one_point': one_point_crossover,
    'uniform':   uniform_crossover,
}


# ─────────────────────────────────────────────────────────────
# SECTION 4: ALGORITM GENETIC
# ─────────────────────────────────────────────────────────────

class GACommunityDetection:
    """
    Algoritm Genetic pentru detectia comunitatilor.

    Codificare: Locus-based adjacency (Pizzuti, 2017)
    Selectie:   Turneu
    Incrucisare: Intr-un punct sau uniforma
    Mutatie:    Inlocuire cu vecin aleator
    Elitism:    Pastram cei mai buni `elite_size` indivizi
    """

    def __init__(
        self,
        graph,
        pop_size=100,
        generations=200,
        crossover_rate=0.8,
        mutation_rate=0.05,
        tournament_size=3,
        elite_size=2,
        fitness_fn='modularity',
        crossover_type='one_point',
        verbose=True,
    ):
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.n = len(self.nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.neighbors = build_neighbor_index(graph, self.nodes, self.node_to_idx)

        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.verbose = verbose

        self.fitness_name = fitness_fn
        base_fn = FITNESS_FUNCTIONS.get(fitness_fn, fitness_modularity)
        self._eval = lambda c: base_fn(c, self.graph, self.nodes)

        self.crossover_fn = CROSSOVER_FUNCTIONS.get(crossover_type, one_point_crossover)

        self.best_fitness_history = []
        self.avg_fitness_history  = []

    def _evaluate_all(self, population):
        return [self._eval(c) for c in population]

    def run(self):
        """Ruleaza algoritmul genetic. Returneaza dict cu rezultate."""
        start = time.time()

        # Initializare populatie
        pop = [random_chromosome(self.n, self.neighbors) for _ in range(self.pop_size)]
        fits = self._evaluate_all(pop)

        best_idx  = int(np.argmax(fits))
        best_chrom = pop[best_idx][:]
        best_fit   = fits[best_idx]

        self.best_fitness_history = [best_fit]
        self.avg_fitness_history  = [float(np.mean(fits))]

        for gen in range(self.generations):
            new_pop = []

            # Elitism
            elite_idx = sorted(range(len(fits)), key=lambda x: fits[x], reverse=True)
            for ei in elite_idx[:self.elite_size]:
                new_pop.append(pop[ei][:])

            # Generare descendenti
            while len(new_pop) < self.pop_size:
                p1 = tournament_selection(pop, fits, self.tournament_size)
                p2 = tournament_selection(pop, fits, self.tournament_size)

                c1, c2 = self.crossover_fn(p1, p2, self.crossover_rate)
                c1 = mutate(c1, self.neighbors, self.mutation_rate)
                c2 = mutate(c2, self.neighbors, self.mutation_rate)
                c1 = repair(c1, self.neighbors)
                c2 = repair(c2, self.neighbors)

                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)

            pop  = new_pop[:self.pop_size]
            fits = self._evaluate_all(pop)

            gen_best = int(np.argmax(fits))
            if fits[gen_best] > best_fit:
                best_fit   = fits[gen_best]
                best_chrom = pop[gen_best][:]

            self.best_fitness_history.append(best_fit)
            self.avg_fitness_history.append(float(np.mean(fits)))

            if self.verbose and (gen + 1) % 25 == 0:
                n_comm = len(set(decode_chromosome(best_chrom)))
                print(f"    Gen {gen+1:4d}/{self.generations} | "
                      f"Best {self.fitness_name}={best_fit:.4f} | "
                      f"Avg={self.avg_fitness_history[-1]:.4f} | "
                      f"Comunitati={n_comm}")

        elapsed   = time.time() - start
        partition = chromosome_to_partition(best_chrom, self.nodes)

        try:
            modularity = nx.community.modularity(self.graph, partition)
        except Exception:
            modularity = float('nan')

        return {
            'algorithm':            f'GA ({FITNESS_DESCRIPTIONS[self.fitness_name]})',
            'partition':            partition,
            'num_communities':      len(partition),
            'fitness':              best_fit,
            'fitness_name':         self.fitness_name,
            'modularity':           modularity,
            'time':                 elapsed,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history':  self.avg_fitness_history,
        }


# ─────────────────────────────────────────────────────────────
# SECTION 5: ALGORITMI PREDEFINITI (NetworkX)
# ─────────────────────────────────────────────────────────────

BUILTIN_ALGORITHMS = {
    'greedy_modularity':  'Greedy Modularity (NetworkX)',
    'label_propagation':  'Label Propagation (NetworkX)',
    'girvan_newman':      'Girvan-Newman (NetworkX)',
    'louvain':            'Louvain (NetworkX >= 3.3)',
}


def run_builtin_algorithm(graph, algorithm='greedy_modularity'):
    """Ruleaza un algoritm predefinit de detectie a comunitatilor."""
    start = time.time()
    partition = None

    if algorithm == 'greedy_modularity':
        partition = list(nx.community.greedy_modularity_communities(graph))

    elif algorithm == 'label_propagation':
        partition = list(nx.community.label_propagation_communities(graph))

    elif algorithm == 'girvan_newman':
        comp = nx.community.girvan_newman(graph)
        # Opreste-te la prima impartire care da modularitate maxima (primele 3 niveluri)
        best_partition = None
        best_q = -1.0
        for _ in range(3):
            try:
                candidate = list(next(comp))
                q = nx.community.modularity(graph, candidate)
                if q > best_q:
                    best_q = q
                    best_partition = candidate
            except StopIteration:
                break
        partition = best_partition or []

    elif algorithm == 'louvain':
        try:
            partition = list(nx.community.louvain_communities(graph, seed=42))
        except AttributeError:
            print("    Louvain nu este disponibil in aceasta versiune de NetworkX. Folosim Greedy Modularity.")
            partition = list(nx.community.greedy_modularity_communities(graph))

    else:
        partition = list(nx.community.greedy_modularity_communities(graph))

    elapsed = time.time() - start

    try:
        modularity = nx.community.modularity(graph, partition)
    except Exception:
        modularity = float('nan')

    return {
        'algorithm':       BUILTIN_ALGORITHMS.get(algorithm, algorithm),
        'partition':       partition,
        'num_communities': len(partition),
        'modularity':      modularity,
        'time':            elapsed,
    }


# ─────────────────────────────────────────────────────────────
# SECTION 6: VIZUALIZARE
# ─────────────────────────────────────────────────────────────

PALETTE = [
    '#e63946', '#457b9d', '#2a9d8f', '#e9c46a', '#f4a261',
    '#6a0572', '#118ab2', '#06d6a0', '#ffd166', '#ef476f',
    '#8ecae6', '#023047', '#ffb703', '#fb8500', '#219ebc',
    '#a8dadc', '#457b9d', '#1d3557', '#48cae4', '#0096c7',
]


def community_colors(partition, nodes):
    """Returneaza culoarea fiecarui nod in functie de comunitate."""
    node_color = {}
    for comm_id, community in enumerate(partition):
        color = PALETTE[comm_id % len(PALETTE)]
        for node in community:
            node_color[node] = color
    return [node_color.get(n, '#aaaaaa') for n in nodes]


def visualize_graph(graph, partition, title="Community Detection", ax=None):
    """Deseneaza graful cu comunitatile colorate."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))

    nodes = list(graph.nodes())
    n = len(nodes)

    colors = community_colors(partition, nodes)

    if n > 500:
        pos = nx.spring_layout(graph, seed=42, k=1.5 / (n ** 0.5))
        node_size = 5
        with_labels = False
    elif n > 100:
        pos = nx.kamada_kawai_layout(graph)
        node_size = 30
        with_labels = False
    else:
        pos = nx.spring_layout(graph, seed=42)
        node_size = max(80, 400 - n * 2)
        with_labels = True

    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.25, edge_color='#888888', width=0.6)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=colors,
                           node_size=node_size, alpha=0.9)
    if with_labels:
        nx.draw_networkx_labels(graph, pos, ax=ax, font_size=6, font_color='white')

    num_comm = len(partition)
    ax.set_title(f"{title}\n{num_comm} comunitati", fontsize=11, fontweight='bold')
    ax.axis('off')


def plot_convergence(best_hist, avg_hist, title="Convergenta GA", save_path=None):
    """Afiseaza curba de convergenta a GA."""
    fig, ax = plt.subplots(figsize=(9, 4))
    gens = range(len(best_hist))
    ax.plot(gens, best_hist, 'b-',  label='Best fitness', linewidth=2)
    ax.plot(gens, avg_hist,  'r--', label='Avg fitness',  linewidth=1.5, alpha=0.7)
    ax.fill_between(gens, avg_hist, best_hist, alpha=0.1, color='blue')
    ax.set_xlabel('Generatie')
    ax.set_ylabel('Fitness')
    ax.set_title(title, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


# ─────────────────────────────────────────────────────────────
# SECTION 7: AFISARE REZULTATE
# ─────────────────────────────────────────────────────────────

def print_graph_info(G, name):
    """Afiseaza statistici de baza ale grafului."""
    print(f"\n{'═'*62}")
    print(f"  Graf: {name}")
    print(f"{'═'*62}")
    print(f"  Noduri:            {G.number_of_nodes()}")
    print(f"  Muchii:            {G.number_of_edges()}")
    print(f"  Densitate:         {nx.density(G):.4f}")
    print(f"  Grad mediu:        {sum(d for _, d in G.degree()) / G.number_of_nodes():.2f}")
    print(f"  Conex:             {nx.is_connected(G)}")
    try:
        cc = nx.average_clustering(G)
        print(f"  Clustering mediu:  {cc:.4f}")
    except Exception:
        pass


def print_result(result, graph_name=""):
    """Afiseaza rezultatul detectiei de comunitati."""
    print(f"\n{'─'*62}")
    print(f"  Algoritm:  {result['algorithm']}")
    if graph_name:
        print(f"  Graf:      {graph_name}")
    print(f"{'─'*62}")
    print(f"  Numar comunitati:   {result['num_communities']}")
    print(f"  Modularitate (Q):   {result.get('modularity', float('nan')):.4f}")
    if result.get('fitness_name') and result['fitness_name'] != 'modularity':
        print(f"  Fitness ({result['fitness_name']}): {result.get('fitness', 'N/A'):.4f}")
    print(f"  Timp executie:      {result['time']:.3f}s")

    sizes = sorted([len(c) for c in result['partition']], reverse=True)
    print(f"  Dimensiuni comunitati: {sizes}")

    print(f"\n  Apartenenta noduri:")
    for cid, comm in enumerate(sorted(result['partition'], key=len, reverse=True)):
        sample = sorted(list(comm))[:8]
        suffix = f"  (+{len(comm)-8} altii)" if len(comm) > 8 else ""
        print(f"    Comunitate {cid+1:2d} ({len(comm):3d} noduri): {sample}{suffix}")


def print_summary(all_results):
    """Afiseaza tabelul sumar final."""
    print(f"\n{'═'*80}")
    print("  SUMAR REZULTATE")
    print(f"{'═'*80}")
    hdr = f"  {'Graf':<20} {'Algoritm':<32} {'Comm':>5} {'Q':>7} {'Timp':>8}"
    print(hdr)
    print(f"  {'-'*76}")
    for graph_name, results in all_results.items():
        for r in results:
            q    = r.get('modularity', float('nan'))
            q_str = f"{q:.4f}" if not (isinstance(q, float) and q != q) else "N/A"
            print(f"  {graph_name:<20} {r['algorithm']:<32} {r['num_communities']:>5} "
                  f"{q_str:>7} {r['time']:>7.2f}s")
    print(f"{'═'*80}")


# ─────────────────────────────────────────────────────────────
# SECTION 8: INCARCARE DATE
# ─────────────────────────────────────────────────────────────

def load_graph(filepath):
    """Incarca un graf din fisier (GML, GraphML, edgelist etc.)."""
    ext = os.path.splitext(filepath)[1].lower()
    loaders = {
        '.gml':     nx.read_gml,
        '.graphml': nx.read_graphml,
        '.xml':     nx.read_graphml,
        '.edgelist':nx.read_edgelist,
        '.net':     nx.read_pajek,
        '.pajek':   nx.read_pajek,
    }
    loader = loaders.get(ext)
    if loader is None:
        raise ValueError(f"Format necunoscut: {ext}")

    G = loader(filepath)
    if G.is_directed():
        G = G.to_undirected()
    G.remove_edges_from(nx.selfloop_edges(G))

    if not nx.is_connected(G):
        lcc = max(nx.connected_components(G), key=len)
        G   = G.subgraph(lcc).copy()
        print(f"  [!] Graf neconex – se foloseste componenta conexa cea mai mare "
              f"({G.number_of_nodes()} noduri, {G.number_of_edges()} muchii).")

    # Reindexare la numere intregi daca e necesar
    if not all(isinstance(n, int) for n in G.nodes()):
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)

    return G


# ─────────────────────────────────────────────────────────────
# SECTION 9: MAIN
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Community Detection cu Algoritmi Genetici – Lab 11 MPP',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument('--graph',       type=str,   default=None,
                   help='Fisier GML de incarcat')
    p.add_argument('--graphs_dir',  type=str,   default='.',
                   help='Director cu fisiere GML (folosit daca --graph nu e specificat)')
    p.add_argument('--builtin',     type=str,   default='greedy_modularity',
                   choices=list(BUILTIN_ALGORITHMS.keys()),
                   help='Algoritm predefinit din NetworkX')
    p.add_argument('--fitness',     type=str,   default='modularity',
                   choices=list(FITNESS_FUNCTIONS.keys()),
                   help='Functia de fitness pentru GA\n'
                        '  modularity          – Modularitate Q (default)\n'
                        '  conductance         – Conductanta medie negativa (BONUS)\n'
                        '  coverage_performance – Coverage+Performance (BONUS)')
    p.add_argument('--all_fitness', action='store_true',
                   help='Ruleaza GA cu toate cele 3 functii de fitness (bonus complet)')
    p.add_argument('--pop_size',    type=int,   default=100,
                   help='Dimensiunea populatiei GA')
    p.add_argument('--generations', type=int,   default=200,
                   help='Numarul de generatii GA')
    p.add_argument('--crossover_rate', type=float, default=0.8)
    p.add_argument('--mutation_rate',  type=float, default=0.05)
    p.add_argument('--tournament_size',type=int,   default=3)
    p.add_argument('--elite_size',     type=int,   default=2)
    p.add_argument('--crossover_type', type=str,   default='one_point',
                   choices=['one_point', 'uniform'])
    p.add_argument('--seed',           type=int,   default=42)
    p.add_argument('--no_ga',    action='store_true', help='Sari peste GA')
    p.add_argument('--no_plot',  action='store_true', help='Sari peste salvarea graficelor')
    p.add_argument('--output_dir', type=str, default='results',
                   help='Director pentru fisierele de iesire')
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "═"*62)
    print("  Community Detection cu Algoritmi Genetici")
    print("  Lab 11 – MPP | Pizzuti (2017)")
    print("═"*62)
    print(f"  NetworkX version: {nx.__version__}")

    # ── Gaseste fisierele de graf ──────────────────────────────
    graph_data = []
    if args.graph:
        name = os.path.splitext(os.path.basename(args.graph))[0]
        print(f"\nIncarcare: {args.graph} ...")
        G = load_graph(args.graph)
        graph_data.append((name, G))
    else:
        gml_files = [
            f for f in os.listdir(args.graphs_dir)
            if f.endswith(('.gml', '.graphml', '.edgelist', '.net'))
        ]
        for fname in sorted(gml_files):
            path = os.path.join(args.graphs_dir, fname)
            name = os.path.splitext(fname)[0]
            print(f"\nIncarcare: {path} ...")
            try:
                G = load_graph(path)
                graph_data.append((name, G))
            except Exception as e:
                print(f"  Eroare: {e}")

    if not graph_data:
        print("\nNu s-au gasit fisiere GML. Se foloseste Karate Club (demo).")
        G = nx.karate_club_graph()
        graph_data.append(('karate_club', G))

    all_results = {}

    for graph_name, G in graph_data:
        print_graph_info(G, graph_name)
        results_for_graph = []

        # ── 1. Algoritm predefinit ────────────────────────────
        print(f"\n[1] Algoritm predefinit: {args.builtin}")
        try:
            res = run_builtin_algorithm(G, args.builtin)
            print_result(res, graph_name)
            results_for_graph.append(res)
        except Exception as e:
            print(f"  Eroare algoritm predefinit: {e}")

        # ── 2. Algoritm Genetic ───────────────────────────────
        if not args.no_ga:
            fitness_list = (
                list(FITNESS_FUNCTIONS.keys()) if args.all_fitness
                else [args.fitness]
            )

            for fit_name in fitness_list:
                print(f"\n[GA] Fitness: {FITNESS_DESCRIPTIONS[fit_name]}")
                print(f"     pop={args.pop_size}, gen={args.generations}, "
                      f"cr={args.crossover_rate}, mr={args.mutation_rate}")
                try:
                    ga = GACommunityDetection(
                        graph=G,
                        pop_size=args.pop_size,
                        generations=args.generations,
                        crossover_rate=args.crossover_rate,
                        mutation_rate=args.mutation_rate,
                        tournament_size=args.tournament_size,
                        elite_size=args.elite_size,
                        fitness_fn=fit_name,
                        crossover_type=args.crossover_type,
                        verbose=True,
                    )
                    ga_res = ga.run()
                    print_result(ga_res, graph_name)
                    results_for_graph.append(ga_res)

                    # Salvare convergenta
                    if not args.no_plot:
                        conv_path = os.path.join(
                            args.output_dir,
                            f"convergence_{graph_name}_{fit_name}.png"
                        )
                        plot_convergence(
                            ga_res['best_fitness_history'],
                            ga_res['avg_fitness_history'],
                            title=f"Convergenta GA – {graph_name} | {FITNESS_DESCRIPTIONS[fit_name]}",
                            save_path=conv_path,
                        )
                        plt.close('all')
                        print(f"\n  Grafic convergenta salvat: {conv_path}")

                except Exception as e:
                    print(f"  Eroare GA: {e}")
                    import traceback; traceback.print_exc()

        all_results[graph_name] = results_for_graph

        # ── Vizualizare comunitati ────────────────────────────
        if not args.no_plot and results_for_graph:
            n_plots = len(results_for_graph)
            fig, axes = plt.subplots(1, n_plots, figsize=(9 * n_plots, 7),
                                     facecolor='#f8f8f8')
            if n_plots == 1:
                axes = [axes]

            for ax, res in zip(axes, results_for_graph):
                q_val = res.get('modularity', float('nan'))
                q_str = f"{q_val:.3f}" if q_val == q_val else "N/A"
                visualize_graph(
                    G, res['partition'],
                    title=f"{res['algorithm']}\nQ={q_str}",
                    ax=ax,
                )

            plt.suptitle(f"Community Detection – {graph_name}", fontsize=14, fontweight='bold')
            plt.tight_layout()
            out_path = os.path.join(args.output_dir, f"communities_{graph_name}.png")
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close('all')
            print(f"\n  Vizualizare comunitati salvata: {out_path}")

    # ── Sumar final ───────────────────────────────────────────
    print_summary(all_results)
    print(f"\nFisierele de iesire se gasesc in directorul: {os.path.abspath(args.output_dir)}/")
    print("Gata!\n")


if __name__ == '__main__':
    main()
