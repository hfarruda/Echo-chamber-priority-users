import igraph as ig
import random
import sys
import numpy as np
import doces as od
import measures
from tqdm import tqdm
import matplotlib.pyplot as plt
import xnetwork as xnet #To save the network (to install it use pip install xnetwork)
import os
import pickle
from matplotlib.gridspec import GridSpec
import scipy.stats as st
from scipy.stats import gaussian_kde
import matplotlib.cm as cm

plt.rcParams.update({
    'figure.figsize': (6,4), 
    "font.size" : 16,
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "STIX",
    "mathtext.fontset": "stix",
    'text.latex.preamble': (            # LaTeX preamble
        r'\usepackage{amsmath,lmodern}'
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


def plot_map(b, b_neighbors, name_out):
    xmin, xmax = -1., 1.
    ymin, ymax = -1., 1.

    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:0.01, ymin:ymax:0.01]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([b, b_neighbors])
    kernel = st.gaussian_kde(values)
    f_ = np.reshape(kernel(positions).T, xx.shape)

    fig = plt.figure(figsize=(3.3,3))
    gs = GridSpec(4,4)
    ax_joint = fig.add_subplot(gs[1:4,0:3])
    ax_marg_x = fig.add_subplot(gs[0,0:3])
    ax_marg_y = fig.add_subplot(gs[1:4,3])

    extent = [-1,1,-1,1]
    ax_joint.imshow(f_.T, extent=extent, origin='lower', cmap=cm.inferno)
    ax_joint.set_ylabel(r"$\hat{b}_{NN}$")
    ax_joint.set_xlabel(r"$\hat{b}$")

    x_grid = np.linspace(-1,1,150)
    kde = gaussian_kde(b, bw_method=0.25)
    y = kde.evaluate(x_grid)
    ax_marg_x.plot(x_grid, y, color = "#404040")
    ax_marg_x.set_xlim(-1,1)

    kde = gaussian_kde(b_neighbors, bw_method=0.25)
    y = kde.evaluate(x_grid)
    ax_marg_y.plot(y, x_grid, color = "#404040")
    ax_marg_y.set_ylim(-1,1)

    ax_marg_x.xaxis.set_visible(False)
    ax_marg_x.yaxis.set_visible(False)
    ax_marg_x.spines['right'].set_visible(False)
    ax_marg_x.spines['left'].set_visible(False)
    ax_marg_x.spines['top'].set_visible(False)

    ax_marg_y.xaxis.set_visible(False)
    ax_marg_y.yaxis.set_visible(False)
    ax_marg_y.spines['right'].set_visible(False)
    ax_marg_y.spines['bottom'].set_visible(False)
    ax_marg_y.spines['top'].set_visible(False)

    plt.tight_layout()

    pos1 = ax_joint.get_position(original=False)
    pos0 = ax_marg_x.get_position(original=False)
    ax_marg_x.set_position([pos1.x0, pos0.y0, pos1.width, pos0.height])

    pos1 = ax_joint.get_position(original=False)
    pos0 = ax_marg_y.get_position(original=False)
    ax_marg_y.set_position([pos0.x0, pos1.y0, pos0.width, pos1.height])

    plt.savefig(name_out)
    plt.close()


def simulate_dynamics(vertex_count, edges, directed, positions, steps, b, stubborn, receiving_filter = [], verbose = False, posting_filter = []):
    simulator = od.Opinion_dynamics(vertex_count = vertex_count, edges=edges, directed=directed, verbose=verbose)
    if len(receiving_filter) != 0:    
        simulator.set_receiving_filter(receiving_filter)
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
    not_verified_pos = set(list(range(len(b_out)))).difference(positions)
    not_verified_pos = np.array(list(not_verified_pos))
    g_out.vs['verified'] = ['Yes' if i in positions else 'No' for i in range(g_out.vcount())]

    bc = measures.bimodality_index(b_out[not_verified_pos])
    return bc, g_out, b_out[not_verified_pos], b_neighbors[not_verified_pos]


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
# seed = 450916448118208508
# random.seed(seed)
# seed_np = random.randint(0,2**32 - 1)
# np.random.seed(seed_np)

small_percentages = np.linspace(0,2,11)
percentages = np.linspace(0,50,26, dtype=int)
percentages = np.concatenate((small_percentages, percentages[2:]))

out_path = "./out/"
create_directory_if_not_exists(out_path)
in_path = "./data/"
create_directory_if_not_exists(in_path)

bc = []
bc_ideologue = []
bc_inset = []
bc_ideologue_inset = []

if os.path.exists(os.path.join(in_path, 'bc_fig7.pickle')) and os.path.exists(os.path.join(in_path, 'bc_ideologue_fig7.pickle')):
    with open(os.path.join(in_path, 'bc_fig7.pickle'), 'rb') as handle:
        bc = pickle.load(handle)
    with open(os.path.join(in_path, 'bc_ideologue_fig7.pickle'), 'rb') as handle:
        bc_ideologue = pickle.load(handle)
else:
    for i in tqdm(range(2)):
        g = ig.Graph()
        g = g.Erdos_Renyi(vertex_count, p)
        g = g.components(mode=ig.WEAK).giant()
        g = g.simplify()
        edges = g.get_edgelist()
        #shuffling directed edges
        order = [np.random.permutation((0,1)) for _ in range(len(edges))]
        edges = [(edge[pos[0]],edge[pos[1]]) for edge, pos in zip(edges, order)]
        edges = np.array(edges)

        bc_iteration = []
        bc_ideologue_iteration = []

        for percentage in percentages:
            posting_filter = np.full(g.vcount(), od.COSINE)
            receiving_filter = np.full(g.vcount(), od.COSINE)
            positions = select_randomly(len(receiving_filter), percentage)
            posting_filter[positions] = od.HALF_COSINE 

            b = np.random.random(size=vertex_count)
            b = b * 2 - 1

            b[positions] = 0
            stubborn = np.zeros(g.vcount())
            stubborn[positions] = 1

            b_not_verified, g_out, b_out, b_neighbors = simulate_dynamics(vertex_count, edges, directed, positions, steps, b, stubborn)

            bc_iteration.append(b_not_verified)

            if i == 0:
                g_out.vs['name'] = [str(i) for i in range(g_out.vcount())]
                if percentage == 2.:
                    plot_map(b_out, b_neighbors, os.path.join(out_path, 'fig7b.pdf'))
                    xnet.igraph2xnet(g_out, os.path.join(in_path, "fig7b.xnet"))
                elif percentage == 20.:
                    plot_map(b_out, b_neighbors, os.path.join(out_path, 'fig7c.pdf'))
                    xnet.igraph2xnet(g_out, os.path.join(in_path, "fig7c.xnet"))

            #Variation
            posting_filter[positions] = od.HALF_COSINE
            receiving_filter[positions] = od.UNIFORM
            b_not_verified, g_out, b_out, b_neighbors = simulate_dynamics(vertex_count, edges, directed, positions, steps, b, stubborn, receiving_filter)
            bc_ideologue_iteration.append(b_not_verified)

            if i == 0:
                g_out.vs['name'] = [str(i) for i in range(g_out.vcount())]
                if percentage == 2.:
                    plot_map(b_out, b_neighbors, os.path.join(out_path, 'fig7d.pdf'))
                    xnet.igraph2xnet(g_out, os.path.join(in_path, "fig7d.xnet"))
                elif percentage == 20.:
                    plot_map(b_out, b_neighbors, os.path.join(out_path, 'fig7e.pdf'))
                    xnet.igraph2xnet(g_out, os.path.join(in_path, "fig7e.xnet"))

        bc.append(bc_iteration)
        bc_ideologue.append(bc_ideologue_iteration)

    #save pickle here
    with open(os.path.join(in_path, 'bc_fig7.pickle'), 'wb') as handle:
        pickle.dump(bc, handle)
    with open(os.path.join(in_path, 'bc_ideologue_fig7.pickle'), 'wb') as handle:
        pickle.dump(bc_ideologue, handle)

fractions = percentages/100.

diag_mean = np.mean(bc, axis=0)
diag_std = np.std(bc, axis=0)

plt.figure()
plt.plot(fractions, diag_mean,color = "#e05d2f")#"#66c2a5")

diag_b_mean = np.mean(bc_ideologue, axis=0)
diag_b_std = np.std(bc_ideologue, axis=0)
plt.plot(fractions, diag_b_mean, color = "#1cb7b3")

plt.legend(["stubborn ", "ideologue"], loc="upper right", frameon=False)

plt.fill_between(fractions, diag_mean - diag_std, diag_mean + diag_std, alpha = 0.4, color = "#e05d2f", linewidth=0)
plt.fill_between(fractions, diag_b_mean - diag_b_std, diag_b_mean + diag_b_std, alpha = 0.4, color = "#1cb7b3", linewidth=0)

ax = plt.gca()
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

plt.hlines(5./9., 0, 0.5, colors='#404040', linestyles='dashed')
plt.ylabel(r"$BC(\hat{b})$")
plt.xlabel(r"fraction of users $u$ with special behavior")
plt.xlim(0, .25)
plt.ylim(0, 1)

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(out_path, "fig7a.pdf"))
plt.close()



g_out = xnet.xnet2igraph(os.path.join(in_path, "fig7b.xnet"))
b = np.array(g_out.vs['b'])
b_neighbors = np.array(measures.average_neighbors_opinion(b, g_out))
verified = np.array(g_out.vs['verified'])
positions = np.array([index for index, value in enumerate(verified) if value == "No"])
not_verified_pos = set(list(range(len(b)))).difference(positions)
b = b[positions]
b_neighbors = b_neighbors[positions]
plot_map(b, b_neighbors, os.path.join(out_path, 'fig7b.pdf'))

                
g_out = xnet.xnet2igraph(os.path.join(in_path, "fig7c.xnet"))
b = np.array(g_out.vs['b'])
b_neighbors = np.array(measures.average_neighbors_opinion(b, g_out))
verified = np.array(g_out.vs['verified'])
positions = np.array([index for index, value in enumerate(verified) if value == "No"])
not_verified_pos = set(list(range(len(b)))).difference(positions)
b = b[positions]
b_neighbors = b_neighbors[positions]
plot_map(b, b_neighbors, os.path.join(out_path, 'fig7c.pdf'))


g_out = xnet.xnet2igraph(os.path.join(in_path, "fig7d.xnet"))
b = np.array(g_out.vs['b'])
b_neighbors = np.array(measures.average_neighbors_opinion(b, g_out))
verified = np.array(g_out.vs['verified'])
positions = np.array([index for index, value in enumerate(verified) if value == "No"])
not_verified_pos = set(list(range(len(b)))).difference(positions)
b = b[positions]
b_neighbors = b_neighbors[positions]
plot_map(b, b_neighbors, os.path.join(out_path, 'fig7d.pdf'))


g_out = xnet.xnet2igraph(os.path.join(in_path, "fig7e.xnet"))
b = np.array(g_out.vs['b'])
b_neighbors = np.array(measures.average_neighbors_opinion(b, g_out))
verified = np.array(g_out.vs['verified'])
positions = np.array([index for index, value in enumerate(verified) if value == "No"])
not_verified_pos = set(list(range(len(b)))).difference(positions)
b = b[positions]
b_neighbors = b_neighbors[positions]
plot_map(b, b_neighbors, os.path.join(out_path, 'fig7e.pdf'))
