import igraph as ig
import random
import sys
import numpy as np
import doces as od
import measures
from tqdm import tqdm
import matplotlib.pyplot as plt
import xnetwork as xnet
import os
import pickle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams.update({
    'figure.figsize': (6,4), 
    "font.size" : 16,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "STIX",
    "mathtext.fontset": "stix",
    'text.latex.preamble': (            # LaTeX preamble
        r'\usepackage{amsmath,lmodern}'
        # ... more packages if needed
    )
})

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


def simulate_dynamics(vertex_count, edges, directed, positions, steps, b, stubborn, receiving_filter = [], verbose = False, posting_filter = []):
    simulator = od.Opinion_dynamics(vertex_count = vertex_count, edges=edges, directed=directed, verbose=verbose)
    if len(receiving_filter) != 0:    
        simulator.set_receiving_filter(receiving_filter)#algorithm
        receiving_filter_tyep = od.CUSTOM
    else:
        receiving_filter_tyep = od.COSINE
    
    if len(posting_filter) != 0:
        simulator.set_posting_filter(posting_filter)
        posting_filter_type = od.CUSTOM
    else:
        posting_filter_type = od.COSINE
    
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
    b_neighbors = measures.average_neighbors_opinion(b_out, g_out)
    b_neighbors = np.array(b_neighbors)
    # new_x, new_y = measures.rotate(b_out, b_neighbors)
    not_verified_pos = set(list(range(len(b_out)))).difference(positions)
    not_verified_pos = np.array(list(not_verified_pos))
    # b_not_verified_diag = new_x[not_verified_pos]
    g_out.vs['verified'] = ['Yes' if i in positions else 'No' for i in range(g_out.vcount())]

    number_of_edges = g_out.ecount()
    delete = np.linspace(0,g_out.vcount()-1, g_out.vcount(),dtype=int)
    stubborn = stubborn.astype(bool)
    delete = list(delete[stubborn])
    g_out.delete_vertices(delete)
    number_of_not_stubborn_edges = g_out.ecount()
    number_of_stubborn_edges = number_of_edges - number_of_not_stubborn_edges
    fraction_of_stubborn_connections = number_of_stubborn_edges / number_of_edges

    return fraction_of_stubborn_connections, g_out, b_out[not_verified_pos], b_neighbors[not_verified_pos]


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

#setting random seed
seed = 450916448118208508
random.seed(seed)
seed_np = random.randint(0,2**32 - 1)
np.random.seed(seed_np)

small_percentages = np.linspace(0,2,11)
percentages = np.linspace(0,50,26, dtype=int)
percentages = np.concatenate((small_percentages, percentages[2:]))
# percentages = np.linspace(0,50,21, dtype=int)

frac_edges = []
frac_edges_aligned = []
frac_edges_no_aligned = []

has_frac_edges = False
has_frac_edges_aligned = False
has_frac_edges_no_aligned = False

if  os.path.exists(os.path.join(in_path,'frac_edges_s3.pickle')):
    with open(os.path.join(in_path,'frac_edges_s3.pickle'), 'rb') as handle:
        frac_edges = pickle.load(handle)
    has_frac_edges = True

if  os.path.exists(os.path.join(in_path,'frac_edges_aligned_s3.pickle')):  
    with open(os.path.join(in_path,'frac_edges_aligned_s3.pickle'), 'rb') as handle:
        frac_edges_aligned = pickle.load(handle)
    has_frac_edges_aligned = True

if  os.path.exists(os.path.join(in_path,'frac_edges_not_aligned_s3.pickle')):  
    with open(os.path.join(in_path,'frac_edges_not_aligned_s3.pickle'), 'rb') as handle:
        frac_edges_no_aligned = pickle.load(handle)
    has_frac_edges_no_aligned = True 

already_done = False
if has_frac_edges and has_frac_edges_aligned and has_frac_edges_no_aligned:
    already_done = True

if not already_done:
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

        frac_edges_iteration = []
        frac_edges_aligned_iteration = []
        frac_edges_no_aligned_iteration = []

        for percentage in percentages:
            posting_filter = np.full(g.vcount(), od.COSINE)
            receiving_filter = np.full(g.vcount(), od.COSINE)
            positions = select_randomly(len(receiving_filter), percentage)

            b = np.random.random(size=vertex_count)
            b = b * 2 - 1

            opinions_aux = [-1,1] * (g.vcount()//2 + 1)
            opinions_aux = np.array(opinions_aux)
            b[positions] = opinions_aux[0:len(positions)]
            stubborn = np.zeros(g.vcount())
            stubborn[positions] = 1

            if not has_frac_edges:
                posting_filter[positions] = od.HALF_COSINE
                fraction_of_stubborn_connections, g_out, b_out, b_neighbors = simulate_dynamics(vertex_count, edges, directed, positions, steps, b, stubborn, posting_filter = posting_filter)
                frac_edges_iteration.append(fraction_of_stubborn_connections)

            #Variation
            if not has_frac_edges_aligned:
                posting_filter[positions] = od.HALF_COSINE
                receiving_filter[positions] = od.UNIFORM
                fraction_of_stubborn_connections, g_out, b_out, b_neighbors = simulate_dynamics(vertex_count, edges, directed, positions, steps, b, stubborn, receiving_filter = receiving_filter, posting_filter = posting_filter)
                frac_edges_aligned_iteration.append(fraction_of_stubborn_connections)

            if not has_frac_edges_no_aligned:
                posting_filter[positions] = od.COSINE
                receiving_filter[positions] = od.UNIFORM
                fraction_of_stubborn_connections, g_out, b_out, b_neighbors = simulate_dynamics(vertex_count, edges, directed, positions, steps, b, stubborn, receiving_filter = receiving_filter, posting_filter = posting_filter)
                frac_edges_no_aligned_iteration.append(fraction_of_stubborn_connections)

        if not has_frac_edges:
            frac_edges.append(frac_edges_iteration)
        if not has_frac_edges_aligned:
            frac_edges_aligned.append(frac_edges_aligned_iteration)
        if not has_frac_edges_no_aligned:
            frac_edges_no_aligned.append(frac_edges_no_aligned_iteration)

    #save pickle here
    with open(os.path.join(in_path,'frac_edges_s3.pickle'), 'wb') as handle:
        pickle.dump(frac_edges, handle)
    with open(os.path.join(in_path,'frac_edges_aligned_s3.pickle'), 'wb') as handle:
        pickle.dump(frac_edges_aligned, handle)
    with open(os.path.join(in_path,'frac_edges_not_aligned_s3.pickle'), 'wb') as handle:
        pickle.dump(frac_edges_no_aligned, handle)

fractions = percentages/100.

# plt.figure(figsize=(6,4))
diag_mean = np.mean(frac_edges, axis=0)
diag_std = np.std(frac_edges, axis=0)
plt.plot(fractions, diag_mean,color = "goldenrod")#"#66c2a5")

diag_b_mean = np.mean(frac_edges_aligned, axis=0)
diag_b_std = np.std(frac_edges_aligned, axis=0)
plt.plot(fractions, diag_b_mean, color = "#5e4fa2")

diag_stubborn_mean = np.mean(frac_edges_no_aligned, axis=0)
diag_stubborn_std = np.std(frac_edges_no_aligned, axis=0)
plt.plot(fractions, diag_stubborn_mean, color = "#66c2a5")

#outro plot aqui
plt.legend([r"non-priority ", r"ideologue ($P_p^{\text{con}}$)", r"ideologue ($P_p^{\text{ali}}$)"], loc="best", frameon=False)

plt.fill_between(fractions, diag_mean - diag_std, diag_mean + diag_std, alpha = 0.4, color = "goldenrod", linewidth=0)
plt.fill_between(fractions, diag_b_mean - diag_b_std, diag_b_mean + diag_b_std, alpha = 0.4, color = "#5e4fa2", linewidth=0)
plt.fill_between(fractions, diag_stubborn_mean - diag_stubborn_std, diag_stubborn_mean + diag_stubborn_std, alpha = 0.4, color = "#5e4fa2", linewidth=0)

ax = plt.gca()
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

plt.ylabel(r"Fraction of edges linked to stubborn")
plt.xlabel(r"Fraction of stubborn users")
plt.xlim(0, .5)
plt.ylim(0, 1.01)

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(out_path, 'figS3a.pdf'))
plt.close()
