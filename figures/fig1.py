import numpy as np 
from scipy.stats import kurtosis
from scipy.stats import skew
import matplotlib.pyplot as plt 
from scipy.stats import gaussian_kde
import matplotlib.cm as cm
from scipy.stats import kurtosis
from scipy.stats import skew
from matplotlib.gridspec import GridSpec
import scipy.stats as st
import doces #use setup.sh to install it
from measures import *
import igraph as ig
import os

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


vertex_count = 10**4
directed = True
network_name = "er"
steps = 100000000
k = 8
delta = 0.1
phi = 0.785

if (directed):
    k = 2*k
p = k/float(vertex_count)

g = ig.Graph(directed=True)
g = g.Erdos_Renyi(vertex_count, p)

g = g.components(mode=ig.WEAK).giant()
g = g.simplify()
edges = g.get_edgelist()

#shuffling directed edges
order = [np.random.permutation((0,1)) for _ in range(len(edges))]
edges = [(edge[pos[0]],edge[pos[1]]) for edge, pos in zip(edges, order)]
edges = np.array(edges)
b = np.random.random(size=vertex_count)
b = b * 2 - 1

simulator = doces.Opinion_dynamics(vertex_count=vertex_count, edges=edges, directed=directed)

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
bc = bimodality_index(b_out)
b_neighbors = average_neighbors_opinion(b_out, g)
new_x, new_y = rotate(b_out, b_neighbors)
bc_hom = bimodality_index(new_x)

b = g.vs['b']
b_neighbors = average_neighbors_opinion(b, g)

plot_map(b, b_neighbors, os.path.join(out_path, 'fig1a.pdf'))

