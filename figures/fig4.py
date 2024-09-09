import numpy as np
import matplotlib.pyplot as plt
import random
import igraph as ig
import doces
from measures import *
from tqdm import tqdm
import os
import pickle

plt.rcParams.update({
    'figure.figsize': (6,4), 
    "font.size" : 14,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "STIX",
    "mathtext.fontset": "stix"
})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")


def select_randomly(length, percentage):
    number_of_nodes = (length * percentage)//100
    positions = np.zeros(length, dtype = int)
    positions[0:number_of_nodes] = 1
    np.random.shuffle(positions)
    return np.argwhere(positions == 1).T[0]

def simulate_dynamics(vertex_count, edges, directed, steps, b, receiving_filter, verbose, posting_filter = []):
    simulator = doces.Opinion_dynamics(vertex_count = vertex_count, edges=edges, directed=directed, verbose=verbose)
    simulator.set_receiving_filter(receiving_filter)#algorithm
    if len(posting_filter) != 0:
        simulator.set_posting_filter(posting_filter)
        posting_filter_type = doces.CUSTOM
    else:
        posting_filter_type = doces.COSINE

    b_aux = b.copy()
    out = simulator.simulate_dynamics(number_of_iterations = steps,
                                    min_opinion = -1., 
                                    max_opinion = 1.,
                                    phi = 0.785,
                                    delta = delta,
                                    posting_filter = posting_filter_type, 
                                    receiving_filter = doces.CUSTOM,
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
    diagonal = bimodality_index(new_x)
    return diagonal, g_out, b_out, b_neighbors



out_path = "./out/"
create_directory_if_not_exists(out_path)
in_path = "./data/"

verbose = False
directed = True 
k = 8
vertex_count = 10000
delta = 0.1
steps = 100000000

if (directed):
    k = 2*k
p = k/float(vertex_count)

#setting random seed
seed = 509162085084481184
random.seed(seed)
seed_np = random.randint(0,2**32 - 1)
np.random.seed(seed_np)
percentages = np.linspace(0,100,21, dtype=int)

bc_hom_com = []
bc_hom_ali = []

if  os.path.exists(os.path.join(in_path,'bc_hom_com.pickle')):
    with open(os.path.join(in_path,'bc_hom_com.pickle'), 'rb') as handle:
        bc_hom_com = pickle.load(handle)
    with open(os.path.join(in_path,'bc_hom_ali.pickle'), 'rb') as handle:
        bc_hom_ali = pickle.load(handle)
else:
    for i in tqdm(range(100)):
        g = ig.Graph()
        g = g.Erdos_Renyi(vertex_count, p)
        g = g.components(mode=ig.WEAK).giant()
        # g.to_directed()
        g = g.simplify()
        edges = g.get_edgelist()
        #shuffling directed edges
        order = [np.random.permutation((0,1)) for _ in range(len(edges))]
        edges = [(edge[pos[0]],edge[pos[1]]) for edge, pos in zip(edges, order)]
        edges = np.array(edges)

        bc_hom_com_iteration = []
        bc_hom_ali_iteration = []

        for percentage in percentages:
            posting_filter = np.full(g.vcount(), doces.COSINE)
            receiving_filter = np.full(g.vcount(), doces.COSINE)
            positions = select_randomly(len(receiving_filter), percentage)
            receiving_filter[positions] = doces.UNIFORM

            b = np.random.random(size=vertex_count)
            b = b * 2 - 1

            diagonal, g_out, b_out, b_neighbors = simulate_dynamics(vertex_count, edges, directed, steps, b, receiving_filter, verbose)
            bc_hom_com_iteration.append(diagonal)

            posting_filter[positions] = doces.HALF_COSINE
            receiving_filter[positions] = doces.UNIFORM
            diagonal, g_out, b_out, b_neighbors = simulate_dynamics(vertex_count, edges, directed, steps, b, receiving_filter, verbose, posting_filter)
            bc_hom_ali_iteration.append(diagonal)

        bc_hom_com.append(bc_hom_com_iteration)
        bc_hom_ali.append(bc_hom_ali_iteration)


    with open(os.path.join(in_path,'bc_hom_com.pickle'), 'wb') as handle:
        pickle.dump(bc_hom_com, handle)
    with open(os.path.join(in_path,'bc_hom_ali.pickle'), 'wb') as handle:
        pickle.dump(bc_hom_ali, handle)


plt.figure()
percentages = np.array(percentages)/100.
diag_mean = np.mean(bc_hom_com, axis=0)
diag_std = np.std(bc_hom_com, axis=0)
plt.plot(percentages, diag_mean,color = "#1cb7b3")

diag_b_mean = np.mean(bc_hom_ali, axis=0)
diag_b_std = np.std(bc_hom_ali, axis=0)
plt.plot(percentages, diag_b_mean, color = "#187261")

#outro plot aqui
plt.legend([r'$P^{\text{con}}_p$', r'$P^{\text{ali}}_p$'], loc="best", frameon=False)

plt.fill_between(percentages, diag_mean - diag_std, diag_mean + diag_std, alpha = 0.4, color = "#1cb7b3", linewidth=0)
plt.fill_between(percentages, diag_b_mean - diag_b_std, diag_b_mean + diag_b_std, alpha = 0.4, color = "#187261", linewidth=0)

ax = plt.gca()
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

plt.hlines(5./9., 0, 1, colors='#e05d2f', linestyles='dashed')
plt.ylabel(r'$BC_{\text{hom}}(b, b_{NN})$')
plt.xlabel(r'fraction of priority users $r$')
plt.xlim(0, 1)
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(out_path, 'fig4.pdf'))
plt.close()
