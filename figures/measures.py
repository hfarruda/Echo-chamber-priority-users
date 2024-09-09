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
    b = np.array(b)
    for i in range(g.vcount()):
        neighbors = g.neighbors(i, mode= "out")# In the paper they use k-out for derected networks.
        if len(neighbors) != 0:
            out.append(np.mean(b[neighbors]))
        else:
            out.append(b[i])
    return out


def bimodality_index(b):
    n = len(b)
    if n == 0:
        return 0.
    return (np.power(skew(b),2) + 1)/(kurtosis(b) + (3*np.power(n-1,2))/((n-2)*(n-3)))

def bimodality_diagonal(b,g):
    b_neighbors = average_neighbors_opinion(b, g)
    new_x, _ = rotate(b, b_neighbors)
    return bimodality_index(new_x)


def plot_map(b, b_neighbors, name_out):
    xmin, xmax = -1., 1.
    ymin, ymax = -1., 1.

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
    ax_joint.set_ylabel(r"$b_{NN}$")
    ax_joint.set_xlabel(r"$b$")

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

