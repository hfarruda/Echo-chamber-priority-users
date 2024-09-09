import igraph as ig
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pickle
from matplotlib import patches
import doces
from measures import *
import graph_tool.all as gt
from scipy.special import zeta as hurwitz
from numpy.random import default_rng
import matplotlib.patches as patches

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.rcParams.update({
    'figure.figsize': (6,4), 
    "font.size" : 16,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "STIX",
    "mathtext.fontset": "stix"
})
rng = default_rng()


def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")


def select_randomly(length, percentage):
    number_of_nodes = int((length * percentage)//100)
    positions = np.zeros(length, dtype = int)
    positions[0:number_of_nodes] = 1
    np.random.shuffle(positions)
    return np.argwhere(positions == 1).T[0]


def simulate_dynamics(vertex_count, edges, directed, positions_stubborn, steps, b, stubborn, receiving_filter = [], verbose = False, posting_filter = []):
    simulator = doces.Opinion_dynamics(vertex_count = vertex_count, edges=edges, directed=directed, verbose=verbose)
    if len(receiving_filter) != 0:    
        simulator.set_receiving_filter(receiving_filter)#algorithm
        receiving_filter_tyep = doces.CUSTOM
    else:
        receiving_filter_tyep = doces.COSINE
    
    if len(posting_filter) != 0:
        simulator.set_posting_filter(posting_filter)
        posting_filter_type = doces.CUSTOM
    else:
        posting_filter_type = doces.COSINE

    simulator.set_stubborn(stubborn)

    b_aux = b.copy()
    out = simulator.simulate_dynamics(number_of_iterations = steps,
                                    min_opinion = -1., 
                                    max_opinion = 1.,
                                    phi = 0.785,
                                    delta = delta,
                                    posting_filter = posting_filter_type, 
                                    receiving_filter = receiving_filter_tyep,
                                    rewire = True,
                                    b=b_aux)
    
    edges_out = out['edges']
    b_out = out['b']
    g_out = ig.Graph(directed = True)
    g_out.add_vertices(list(range(len(b_out))))
    g_out.add_edges(edges_out)
    g_out.vs['b'] = b_out
    b_out = np.array(g_out.vs['b'])
    b_neighbors = average_neighbors_opinion(b_out, g_out)
    new_x, _ = rotate(b_out, b_neighbors)

    not_verified_pos = set(list(range(len(b_out)))).difference(positions_stubborn)
    not_verified_pos = np.array(list(not_verified_pos))
    b_not_verified_diag = new_x[not_verified_pos]

    diagonal = bimodality_index(b_not_verified_diag)
    return diagonal, g_out, b_out, b_neighbors


def create_power(network_topology, size = 10000, minimum_degree = 3, power_law_exponent = 2.43):
    if network_topology == 'scale-free':
        k_max = 100*size
        k_min = minimum_degree
        gamma = power_law_exponent
        support = np.arange(k_min, k_max)
        mass = (support**-gamma)/hurwitz(gamma,k_min)
        cdf = mass.cumsum()
        g = gt.random_graph(size,lambda: (np.searchsorted(cdf,rng.random(),side='left') + k_min,rng.poisson(np.sum(support*mass))),verbose=False)
    elif network_topology == 'full_scale-free':
        k_max = 100*size
        k_min = minimum_degree
        gamma = power_law_exponent
        support = np.arange(k_min, k_max)
        mass = (support**-gamma)/hurwitz(gamma,k_min)
        cdf = mass.cumsum()
        g = gt.random_graph(size,lambda: (np.searchsorted(cdf,rng.random(),side='left') + k_min, (np.searchsorted(cdf,rng.random(),side='left')) + k_min),verbose=False)
    edgelist = np.array([e for e in g.iter_edges()])
    return edgelist


out_path = "./out/"
create_directory_if_not_exists(out_path)
in_path = "./data/"
create_directory_if_not_exists(in_path)

verbose = False
directed = True 
k = 8
vertex_count = 10000
delta = 0.1
steps = 100000000

if (directed):
    k = 2*k
p = k/float(vertex_count)


for network_type in ['er', 'power']:
    print (network_type)
    small_percentages = np.linspace(0,5,21)
    percentages = np.linspace(0,50,21, dtype=int)

    bc_homs = []
    if network_type == 'er' and os.path.exists(os.path.join(in_path, 'bc_hom_er_figS2.pickle')):
        with open(os.path.join(in_path, 'bc_hom_er_figS2.pickle'), 'rb') as handle:
            bc_homs = pickle.load(handle)
    elif network_type == 'power' and os.path.exists(os.path.join(in_path, 'bc_hom_power_figS2.pickle')):
        with open(os.path.join(in_path, 'bc_hom_power_figS2.pickle'), 'rb') as handle:
            bc_homs = pickle.load(handle)
    else:
        for i in tqdm(range(100)):
            if network_type == 'er':
                g = ig.Graph()
                g = g.Erdos_Renyi(vertex_count, p)
                g = g.components(mode=ig.WEAK).giant()
                g = g.simplify()
                edges = g.get_edgelist()
                vcount = g.vcount()
            elif network_type == 'power':
                edges = create_power('full_scale-free', size=vertex_count)
                edges = np.array(edges)
                vcount = np.max(edges) + 1
            #shuffling directed edges
            order = [np.random.permutation((0,1)) for _ in range(len(edges))]
            edges = [(edge[pos[0]],edge[pos[1]]) for edge, pos in zip(edges, order)]
            edges = np.array(edges)

            bc_homs_iteration = []
            for percentage in percentages:
                posting_filter = np.full(vcount, doces.COSINE)
                receiving_filter = np.full(vcount, doces.COSINE)
                positions = select_randomly(len(receiving_filter), percentage)
                receiving_filter[positions] = doces.UNIFORM
                b = np.random.random(size=vertex_count)
                b = b * 2 - 1
                line = []
                for percentage_stubborn in small_percentages:
                    if percentage_stubborn > percentage:
                        percentage_stubborn = percentage
                    b_aux = b.copy()
                    positions_aux = positions.copy()
                    np.random.shuffle(positions_aux)
                    number_of_nodes = int((g.vcount() * percentage_stubborn)//100)
                    positions_aux = positions_aux[0:number_of_nodes]
                    opinions_aux = [-1,1] * (g.vcount()//2 + 1)
                    opinions_aux = np.array(opinions_aux)
                    b_aux[positions_aux] = opinions_aux[0:len(positions_aux)]
                    stubborn = np.zeros(g.vcount())
                    stubborn[positions_aux] = 1

                    diagonal, g_out, b_out, b_neighbors = simulate_dynamics(vertex_count, edges, directed, positions, steps, b_aux, stubborn, receiving_filter=receiving_filter, posting_filter=posting_filter)

                    line.append(diagonal)

                bc_homs_iteration.append(line)
            
            bc_homs_iteration = np.array(bc_homs_iteration)
            bc_homs.append(bc_homs_iteration)
            
        #save pickle here
        if network_type == 'er':
            with open(os.path.join(in_path, 'bc_hom_er_figS2.pickle'), 'wb') as handle:
                pickle.dump(bc_homs, handle)
        elif network_type == 'power':
            with open(os.path.join(in_path, 'bc_hom_power_figS2.pickle'), 'wb') as handle:
                pickle.dump(bc_homs, handle)

    points = [[0, 0], [.05, .05], [.05, .50], [0, .50]]
    points_crop = [[0, 0], [.05, 0], [.05, .05]]
    patch_crop = patches.Polygon(xy=points_crop, closed=True, facecolor = "w", zorder = 1)
    patch = patches.Polygon(xy=points, closed=True, facecolor = "w", zorder = 0)

    fig, ax = plt.subplots(figsize=(6,4))

    M = np.mean(bc_homs, axis=0)
    x = np.array(small_percentages)/100.
    y = np.array(percentages)/100.

    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)

    extent = [x_min, x_max, y_min, y_max]

    X, Y = np.meshgrid(x, y)

    ax.contourf(X, Y, M, 8, alpha=.75, cmap=plt.cm.hot)
    ax.add_patch(patch_crop)
    ax.add_patch(patch)

    C = plt.contour(X, Y, M, 8, colors='black')
    for c in C.collections:
        c.set_clip_path(patch)

    ax.clabel(C, inline=1, fontsize=14)
    ax.set_xlabel(r"Fraction of stubborn users $s$")
    ax.set_ylabel(r"Fraction of priority users $r$")

    plt.tight_layout()

    if network_type == 'er':
        plt.savefig(os.path.join(out_path, 'figS2a.pdf'))
    elif network_type == 'power':
        plt.savefig(os.path.join(out_path, 'figS2b.pdf'))
    plt.close()