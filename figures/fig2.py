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

def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")

out_path = "./out/"

create_directory_if_not_exists(out_path)

in_path = "./data/"

directed = True
vertex_count = 10000
steps = 100000000
k = 8
delta = 0.1

#setting random seed
seed = 2085084418450916
random.seed(seed)
seed_np = random.randint(0,2**32 - 1)
np.random.seed(seed_np)

phis = np.linspace(0,np.pi,17)

if (directed):
    k = 2*k
p = k/float(vertex_count)

bc = []
bc_hom = []

if  os.path.exists(os.path.join(in_path, 'bc_hom.pickle')):
    with open(os.path.join(in_path, 'phis.pickle'), 'rb') as handle:
        phis = pickle.load(handle)

    with open(os.path.join(in_path,'bc.pickle'), 'rb') as handle:
        bc = pickle.load(handle)

    with open(os.path.join(in_path,'bc_hom.pickle'), 'rb') as handle:
        bc_hom = pickle.load(handle)

else:
    for sample in tqdm(range(100)):
        bc_iteration = []
        bc_hom_iteration = []
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
        b = np.random.random(size=vertex_count)
        b = b * 2 - 1
        for phi in phis:
            simulator = doces.Opinion_dynamics(vertex_count=vertex_count, edges=edges, directed=directed, verbose=False)

            out = simulator.simulate_dynamics(number_of_iterations = steps,
                                        min_opinion = -1., 
                                        max_opinion = 1.,
                                        phi = phi,
                                        delta = delta,
                                        posting_filter = doces.COSINE, 
                                        receiving_filter = doces.COSINE,
                                        rewire = True,
                                        b=b)

            edges_out = out['edges']
            b_out = out['b']
            g = ig.Graph(directed = True)
            g.add_vertices(list(range(len(b_out))))
            g.add_edges(edges_out)
            g.vs['b'] = b_out
            g = g.components(mode=ig.WEAK).giant()
            b_out = np.array(g.vs['b'])
            bimodality = bimodality_index(b_out)
            bc_iteration.append(bimodality)
            b_neighbors = average_neighbors_opinion(b_out, g)
            new_x, new_y = rotate(b_out, b_neighbors)
            diagonal = bimodality_index(new_x)
            bc_hom_iteration.append(diagonal)
            # print(f"bimodality: %.2f, bimodality diagonal: %.2f\n"%(bimodality,diagonal))
            
        bc.append(bc_iteration)
        bc_hom.append(bc_hom_iteration)

    #save pickle
    with open(os.path.join(in_path,'phis.pickle'), 'wb') as handle:
        pickle.dump(phis, handle)

    with open(os.path.join(in_path,'bc.pickle'), 'wb') as handle:
        pickle.dump(bc, handle)

    with open(os.path.join(in_path,'bc_hom.pickle'), 'wb') as handle:
        pickle.dump(bc_hom, handle)
    
plt.figure(figsize=(6,4))
b_mean = np.mean(bc, axis=0)
b_std = np.std(bc, axis=0)

diag_mean = np.mean(bc_hom, axis=0)
diag_std = np.std(bc_hom, axis=0)

plt.plot(phis, b_mean, color = "#8a49c4")
plt.plot(phis, diag_mean, color = "#187261")

plt.legend([r"$b$", r"$b^{\dagger}$"], loc="best", frameon=False)

plt.fill_between(phis, b_mean - b_std, b_mean + b_std, alpha = 0.4,color = "#8a49c4", linewidth=0)
plt.fill_between(phis, diag_mean - diag_std, diag_mean + diag_std, alpha = 0.4,color = "#187261", linewidth=0)

plt.hlines(5./9., 0, np.pi, colors='#404040', linestyles='dashed')

ax = plt.gca()
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

plt.ylabel(r"$BC$")
plt.xlabel(r"$\phi$")
plt.xlim(0, np.pi)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(out_path, 'fig2.pdf'))
plt.close()

