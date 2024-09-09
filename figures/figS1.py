# Code adapted from Hohmann, M., Devriendt, K., & Coscia, M. (2023). Quantifying ideological polarization on a network using generalized Euclidean distance. Science Advances
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import KernelDensity
from scipy.special import binom
from scipy.sparse import csgraph
from tqdm import tqdm
import numpy as np 
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy import stats
import matplotlib.pyplot as plt 
import xnetwork as xnet
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

def create_rotation_matrix(angle):
    matrix = [[np.cos(angle), -np.sin(angle)],
              [np.sin(angle),  np.cos(angle)]]
    matrix = np.array(matrix)
    return matrix

def rotate(x, y, angle=np.pi/4):
    rotation_matrix = create_rotation_matrix(angle)
    m = np.array([x, y]).T
    rotated_m = np.matmul(m, rotation_matrix)
    new_x = rotated_m[:,0]
    new_y = rotated_m[:,1]
    return new_x, new_y

def average_neighbors_opinion(b, g):
    #https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-019-0213-9
    out = []
    for i in g.nodes():
        neighbors = [b[n] for n in g.neighbors(i)]# In the paper they use k-out for derected networks.
        if len(neighbors) != 0:
            out.append(np.mean(neighbors))
        else:
            out.append(b[i])
    return out

def bimodality_index(b):
    if type(b) == type(dict()):
      b = list(b.values())
    n = len(b)
    if n == 0:
        return 0.
    return (np.power(skew(b),2) + 1)/(kurtosis(b) + (3*np.power(n-1,2))/((n-2)*(n-3)))

def bimodality_diagonal(b,g):
    b_neighbors = average_neighbors_opinion(b, g)
    new_x, new_y = rotate(list(b.values()), b_neighbors)
    return bimodality_index(new_x)

def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")


# Function to calculate the pseudoinverse of the Laplacian of the network
def _ge_Q(network):
    A = nx.adjacency_matrix(network).todense().astype(float)
    return np.linalg.pinv(csgraph.laplacian(np.matrix(A), normed=False))

def ge(src, trg, network, Q=None):
    """Calculate GE for network.

    Parameters:
    ----------
    srg: vector specifying node polarities
    trg: vector specifying node polarities
    network: networkx graph
    Q: pseudoinverse of Laplacian of the network
    """
    if nx.number_connected_components(network) > 1:
        raise ValueError("""Node vector distance is only valid if calculated on a network with a single connected component.
                       The network passed has more than one.""")
    src = np.array([src[n] if n in src else 0. for n in network.nodes()])
    trg = np.array([trg[n] if n in trg else 0. for n in network.nodes()])
    diff = src - trg
    if Q is None:
        Q = _ge_Q(network)

    ge_dist = diff.T.dot(np.array(Q).dot(diff))

    if ge_dist < 0:
        ge_dist = 0

    return np.sqrt(ge_dist)


# This function created the 2D KDE cell estimations, for plotting.
def kde2D(x, y, bandwidth, xbins = 100j, ybins = 100j, **kwargs): 
   xx, yy = np.mgrid[-1:1:xbins, -1:1:ybins]
   xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
   xy_train  = np.vstack([y, x]).T
   kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
   kde_skl.fit(xy_train)
   z = np.exp(kde_skl.score_samples(xy_sample))
   return xx, yy, np.reshape(z, xx.shape)

# Utility function to write the 2D KDE to a dataframe (and then to file)
def make_kde(df):
   xx, yy, zz = kde2D(df["pol"], df["avg_neighbor"], 0.1)
   df2 = pd.DataFrame()
   df2["x"] = xx.flatten()
   df2["y"] = yy.flatten()
   df2["z"] = zz.flatten()
   return df2

# This function creates the opinion vector o. It requires two inputs:
# - size: half the number of the nodes in the network;
# - factor: the absolute value of the average polarity of each side (mu in the paper)
# If factor = 0 there is no opinion polarizatio, with factor = 1 opinions cluster at +/- 1
def make_o(size, factor):
   o = np.random.normal(size = size, loc = factor, scale = 0.2) # Make a random normal distribution with std = 0.2
   o[o > 1] = 1 - (o[o > 1] - 1)                                # Mirror values out of the +/-1 bounds to be inbound
   o = np.concatenate([o, -o])                                  # Create the negative side of the distribution
   o.sort()                                                     # Sort, this will create community homophily in the SBM
   return {i: o[i] for i in range(o.shape[0])}                  # Transform into a dictionary, which si the input needed by the function

# This function modifies the community connection probability in the SBM
# Specifically, it sets to zero the probabilities between communities too far
# in the polarity specturm. Takes as input:
# - p: the original SBM connection probability (a CxC matrix)
# - k: how many communities a community connects to. This is n in the main paper.
def update_p(p, k):
   p_sum = p.sum()                                                                           # Save the sum of p entries, this must be constant to ensure same expected # of edges
   for col in range(2):                                                                      # For every column...
      for row in range(2):                                                                   # ...and every row...
         if ((col < 2 - k) and ((row - col) > k)) or ((row < 2 - k) and ((col - row) > k)):  # ...find the entries that are k-1 steps away from the diagonal...
            p[row, col] = 0                                                                  # ...and set them to zero
   p *= (p_sum / p.sum())                                                                    # Make sure that the new p sums to the same value as the old one
   return p

runs = 100

results = []                                         # Stores all numerical results per parameter combination: delta, assortativity, and RWC
factors = (0.0, 0.2, 0.4, 0.6, 0.8)                      # All testes mu values
in_p =  (0.0085,  0.039,  0.054,  0.062,  0.064,  0.067) # All tested p_in values
out_p = (0.0085, 0.0042, 0.0024, 0.0012, 0.0006, 0.0003) # All tested p_out values

df_result = pd.DataFrame(columns=['run', 'in_p',  'out_p',  'conn_neigh',  'factor',  'delta', 'BC', 'BC_hom', 'diameter'])

for run in tqdm(range(runs)):
    for i in range(len(in_p)):                                                       # Loop over all possible p_in-p_out pairs
        probs = np.full((2, 2), out_p[i])                                             # Initialize SBM connection probabilities with p_out
        np.fill_diagonal(probs, in_p[i])                                              # Set diagonal elements of connection probability matrix to be equal to p_in
        # Loop from n = 7 to n = 1
        # Change the matrix of community connection probability p by removing connections between disagreeing communities
        G = nx.stochastic_block_model(sizes = [500] * 2, p = probs)                # Initialize SBM
        while nx.number_connected_components(G) > 1:                               # If the SBM has more than one connected component...
            G = nx.stochastic_block_model(sizes = [500] * 2, p = probs)             # ...reinitialize it, otherwise we cannot compute delta or RWC.
         
        Q = _ge_Q(G)                                                            # Cache the pseudoinverse of the Laplacian, since it's the same regardless of the opinion polarization factor mu
        for factor in factors:                                                     # Loop over all possible values of opinion polarization mu
            o = make_o(len(G.nodes) // 2, factor)                                   # Generate the opinion vector o depending on mu's value (and G's size)
            nx.set_node_attributes(G, o, "polar")                                   # Attach the opinion value to all nodes as an attribute -- required to calculate assortativity
            pol = ge(o, {}, G, Q = Q)   
            diameter = nx.diameter(G)
            bc = bimodality_index(o)
            bc_hom = bimodality_diagonal(o,G)
            df_add = pd.DataFrame({'run':[run], "in_p":[in_p[i]], "out_p":[out_p[i]], "factor":[factor], "delta":[pol], "BC":[bc], "BC_hom":[bc_hom], "diameter": [diameter]})
            df_result = pd.concat([df_result, df_add], ignore_index = True) 
            df_result.reset_index()
            if run == 0:
                g = ig.Graph()
                g.add_vertices(G.nodes)
                g.vs['name'] = [str(i) for i in G.nodes]
                g.vs['b'] = [o[i] for i in G.nodes]
                g.add_edges([e for e in G.edges])
                print(f"avg. degree: {np.mean([G.degree[i] for i in G.nodes])}")

# # Extract the required columns
# pol = df_result[df_result['in_p']==0.0085][df_result['out_p']==0.0085]['delta']
# BC = df_result[df_result['in_p']==0.0085][df_result['out_p']==0.0085]['BC']
# BC_hom = df_result[df_result['in_p']==0.0085][df_result['out_p']==0.0085]['BC_hom']

# # Create a scatter plot with 2 panels (1 line)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# # Plot the scatter plot for 'BC' against 'pol'
# ax1.scatter(pol, BC, color='blue', label='BC', alpha = 0.1)
# ax1.set_title('BC Scatter Plot')
# ax1.set_xlabel('delta')
# ax1.set_ylabel('BC Value')
# ax1.legend()

# pearson = stats.pearsonr(pol, BC).statistic
# spearman = stats.spearmanr(pol, BC).statistic
# print(f"Pearson (BC): {pearson}, Spearman (BC): {spearman}")


# # Plot the scatter plot for 'BC_hom' against 'pol'
# ax2.scatter(pol, BC_hom, color='red', label='BC_hom', alpha = 0.1)
# ax2.set_title('BC_hom Scatter Plot')
# ax2.set_xlabel('delta')
# ax2.set_ylabel('BC_hom Value')
# ax2.legend()
# pearson = stats.pearsonr(pol, BC_hom).statistic
# spearman = stats.spearmanr(pol, BC_hom).statistic
# print(f"Pearson (BC_hom): {pearson}, Spearman (BC_hom): {spearman}")

# # Display the plot
# plt.tight_layout()
# plt.savefig("comparison_2_com.pdf")
# # plt.show()
# plt.close()



in_p =  (0.0085,  0.039,  0.062,  0.067) # All tested p_in values
out_p = (0.0085, 0.0042, 0.0012, 0.0003) # All tested p_out values
colors = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c"]

expect_degree = [500*in_p[i] + 500*out_p[i] for i in range(len(in_p))]

pol_vec = [np.array(df_result[df_result['in_p']==in_p[i]][df_result['out_p']==out_p[i]]['delta']) for i in range(len(in_p))]
BC_hom_vec = [np.array(df_result[df_result['in_p']==in_p[i]][df_result['out_p']==out_p[i]]['BC_hom']) for i in range(len(in_p))]

fig, ax = plt.subplots(1,1)
# Plot the scatter plot for 'BC_hom' against 'pol'
for i in range(len(in_p)):
    pol = pol_vec[i] #df[df['in_p']==in_p[i]][df['out_p']==out_p[i]]['delta']
    # BC = df[df['in_p']==in_p[i]][df['out_p']==out_p[i]]['BC']
    BC_hom = BC_hom_vec[i] #df[df['in_p']==in_p[i]][df['out_p']==out_p[i]]['BC_hom']
    
    ax.scatter(pol, BC_hom, color=colors[i], label=str(expect_degree[i]), alpha = 0.5)
    # ax.set_title('BC_hom Scatter Plot')
    ax.set_xlabel(r'$\delta_{G,b}$')
    ax.set_ylabel(r'$BC_{hom}$')
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    pearson = stats.pearsonr(pol, BC_hom).statistic
    spearman = stats.spearmanr(pol, BC_hom).statistic
    print(f"<k> = {expect_degree[i]}")
    print(f"Pearson (BC_hom): {pearson}, Spearman (BC_hom): {spearman}")

# Display the plot
ax.legend(title = r'avg. degree $\langle k \rangle$', frameon=False)
plt.tight_layout()

out_path = "./out/"
create_directory_if_not_exists(out_path)
plt.savefig(os.path.join(out_path, 'figS1.pdf'))
plt.close()