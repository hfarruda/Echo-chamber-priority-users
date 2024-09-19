import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import skew, kurtosis
from tqdm.auto import tqdm
import graph_tool.all as gt
import doces as od
from joblib import Parallel, delayed

W = 5.8    # Figure width in inches, approximately A4-width - 2*1.25in margin
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

def bimodality_index(b):
    n = len(b)
    return (np.power(skew(b),2) + 1)/(kurtosis(b) + (3*np.power(n-1,2))/((n-2)*(n-3)))

def bc_hom(b,b_NN):
    n = len(b)
    R = np.array([[np.sqrt(2)/2, -np.sqrt(2)/2],[np.sqrt(2)/2, np.sqrt(2)/2]])
    M = np.column_stack((b, b_NN))
    b_dagger = np.matmul(M,R)[:,0]
    return (np.power(skew(b_dagger),2) + 1)/(kurtosis(b_dagger) + (3*np.power(n-1,2))/((n-2)*(n-3)))


def peaks_of_distr(b,cut,no_of_bins):
    custom_bins = np.linspace(0,1,no_of_bins)
    mass, bins = np.histogram(b,bins=custom_bins,density=False)
    cut_index = np.searchsorted(bins,cut)
    first_half = bins[:cut_index][:-1] + (bins[:cut_index][1:] - bins[:cut_index][:-1])/2
    mass_y1 = np.sum(mass[:cut_index-1])/np.sum(mass)
    mass_y2 = np.sum(mass[cut_index-1:])/np.sum(mass)
    if len(first_half) == 0:
        y1 = np.nan
        alpha_y1 = np.nan
    else:
        peak_1_max = np.max(mass[:cut_index-1])
        peak_1_max_index = np.where(mass[:cut_index-1]==peak_1_max)[0][0]
        y1, alpha_y1 = first_half[peak_1_max_index], peak_1_max/np.sum(mass)
    
    second_half = bins[cut_index:][:-1] + (bins[cut_index:][1:] - bins[cut_index:][:-1])/2
    if len(second_half)==0:
        y2 = np.nan
        alpha_y2 = np.nan
    else:
        second_half = np.concatenate((np.array([second_half[0]-((bins[cut_index:][1:] - bins[cut_index:][:-1])/2)[0]]),second_half))
        peak_2_max = np.max(mass[cut_index-1:])
        peak_2_max_index = np.where(mass[cut_index-1:]==peak_2_max)[0][0]
        y2, alpha_y2 = second_half[peak_2_max_index], peak_2_max/np.sum(mass)
    return y1, alpha_y1, mass_y1, y2, alpha_y2, mass_y2


def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")


###
# Progress bar for parallel execution
# from https://stackoverflow.com/a/61900501
class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)
    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)
    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def bimodality_index(b):
    n = len(b)
    return (np.power(skew(b),2) + 1)/(kurtosis(b) + (3*np.power(n-1,2))/((n-2)*(n-3)))


# Function to loop
def nested_simulation(N, edge_list, r_filt, p_filt, percent_of_stubborn_users, verified_users, seed):
    stubborns_are_verified = False
    extreme_zealots = True # Extremist users if true, moderate if false

    rng = np.random.default_rng(seed = seed)
    initial_opinions = 2*rng.random(N) - 1
    simulator = od.Opinion_dynamics(vertex_count=N, edges=edge_list, directed=True, verbose=False)

    # Set stubborn configuration
    if percent_of_stubborn_users > 0:
        is_stubborn = np.full(N, False)

        if stubborns_are_verified:
            no_of_stubborn_users = np.int32(len(verified_users)*percent_of_stubborn_users/10)
            stubborn_users = rng.choice(verified_users, size = no_of_stubborn_users,replace=False)
        else:
            no_of_stubborn_users = np.int32(N*percent_of_stubborn_users/100)
            stubborn_users = rng.choice(np.arange(N), size=no_of_stubborn_users,replace=False)
        
        p_filt[stubborn_users] = od.HALF_COSINE
        # r_filt[stubborn_users] = od.UNIFORM 
        is_stubborn[stubborn_users] = True

        # Set stubborn opinions
        if extreme_zealots:
            len1 = no_of_stubborn_users // 2
            len2 = no_of_stubborn_users - len1
            initial_opinions[stubborn_users] = rng.permutation(np.concatenate((np.full(len1, 1),np.full(len2, -1))))
        else:
            initial_opinions[stubborn_users] = np.zeros(no_of_stubborn_users)
        simulator.set_stubborn(is_stubborn)
    else:
        stubborn_users = np.array([],dtype=np.int32)

    simulator.set_posting_filter(p_filt)

    result = simulator.simulate_dynamics(number_of_iterations = 100000000,
                                min_opinion = -1, 
                                max_opinion = 1,
                                phi = 1.473,
                                delta = 0.1,
                                posting_filter = od.CUSTOM, 
                                receiving_filter = od.COSINE,
                                rewire = True,
                                b=initial_opinions,
                                rand_seed = seed)

    net = gt.Graph()
    net.add_edge_list(result['edges'])
    opinions = np.array(result['b'])
    neighbours_avg_opinion = np.array([np.mean(opinions[net.get_out_neighbors(v)]) for v in net.iter_vertices()])
    return stubborn_users, opinions, neighbours_avg_opinion


# Runtime parameters
runs=5
num_sims = 100 # Each simulation contributes to the mean and std of BC values
rng = np.random.default_rng(seed = 0)

n_jobs = 10 #Number of cores
###

## Parameters
num_points = 51

# Plotting parameters
percent_of_verified = 0 # fixed
stubborn_range = np.linspace(0,5,num_points) 

out_path = "./out/"
create_directory_if_not_exists(out_path)
in_path = "./data/"
create_directory_if_not_exists(in_path)

for run in np.arange(runs)+1:
    array_of_seeds = np.arange(num_sims*num_points)+(run-1)*num_sims*num_points+1
    if not os.path.exists(os.path.join(in_path, f'only_stubborn_extremists_0-5_{num_points}-data_points_run_{run}.pickle')):
        # Load network
        network = gt.load_graph('random_network_z8.gt')
        print('Network loaded with ' + str(network.num_vertices()) + ', mean in-degree ' + str(np.mean(network.get_in_degrees(network.get_vertices()))) + ' and mean out-degree ' + str(np.mean(network.get_out_degrees(network.get_vertices()))))
        edge_list = np.array([e for e in network.iter_edges()])

        ## Configuring functions and user behavior
        posting_filter = np.full(network.num_vertices(), od.COSINE)
        receiving_filter = np.full(network.num_vertices(), od.COSINE)

        # Verified users setup
        if percent_of_verified > 0:
            no_of_verified_users = np.int32(network.num_vertices()*percent_of_verified/100)

            verified_users = rng.choice(np.arange(network.num_vertices()), size=no_of_verified_users,replace=False)

            receiving_filter[verified_users] = od.UNIFORM
        # -----------------------

        data_BC = {}
        # Iteration
        for percent_order, percent_of_stubborn in enumerate(stubborn_range):
            print('PERCENTAGE: ' + str(percent_of_stubborn) + ' out of ' + str(np.max(stubborn_range)))
            sliced_array_of_seeds = array_of_seeds[percent_order*num_sims:percent_order*num_sims+num_sims]
            outputs = ProgressParallel(n_jobs=n_jobs)(delayed(nested_simulation)(network.num_vertices(), edge_list, receiving_filter, posting_filter, percent_of_stubborn, [], sliced_array_of_seeds[sim]) for sim in range(num_sims))
            opinions_matrix = np.column_stack(([el[1] for el in outputs])) #= np.array((network.num_vertices(),num_sims))
            b_NN_matrix = np.column_stack(([el[2] for el in outputs]))
            stubborn_array = np.column_stack(([el[0] for el in outputs]))
            data_BC[percent_of_stubborn] = {'stubborn_users':stubborn_array,'opinions':opinions_matrix,'b_NN':b_NN_matrix}

        with open(os.path.join(in_path, f'only_stubborn_extremists_0-5_{num_points}-data_points_run_{run}.pickle'),'wb') as pickled_data_file:
            pickle.dump(data_BC, pickled_data_file)


mode = 'echo_chamber'
verifieds_up_to = 10
x_ax = np.linspace(0,verifieds_up_to//2,num_points)

data = {}

for run in np.arange(5)+1:
    with open(os.path.join(in_path, f'only_stubborn_extremists_0-5_{num_points}-data_points_run_{run}.pickle'),'rb') as pickled_data:
        data[run-1] = pickle.load(pickled_data)

# 'data' is a dictionary whose keys correspond to the percentages (points in x_ax) and values are a dictionary with two keys:
# 'stubborn_users': np.ndarray of dimension (number of simulations, number of stubborn users)
# 'opinions': np.ndarray of dimension (size of network, number of simulations)

## Plot 1 - mean and standard deviation

mean_bc = []
std_bc = []
all_bc_values = []
quantiles_bc = []
peak1_bc = []
peak1_bc_alphas = []
peak1_bc_mass = []
peak2_bc = []
peak2_bc_alphas = []
peak2_bc_mass = []

for x in x_ax:
    list_of_bcs = []
    #L_skew = []
    #L_kurt = []
    for run in np.arange(runs):
        for sim in np.arange(num_sims):
            influencers = data[run][x]['stubborn_users'][:,sim]
            opinion_distr = data[run][x]['opinions'][:,sim]
            if mode == 'echo_chamber':
                neighbors_avg_opinion = data[run][x]['b_NN'][:,sim]

            if len(influencers) > 0:
                #
                if mode == 'echo_chamber':
                    bc_coeff = bc_hom(np.delete(opinion_distr,influencers), np.delete(neighbors_avg_opinion,influencers)) # remove influencers to calculate
                else:
                    bc_coeff = bimodality_index(np.delete(opinion_distr,influencers)) # remove influencers to calculate
            else:
                #bc_coeff = bimodality_index(opinion_distr) # remove influencers to calculate
                if mode == 'echo_chamber':
                    bc_coeff = bc_hom(opinion_distr, neighbors_avg_opinion)
                else:
                    bc_coeff = bimodality_index(opinion_distr)
            list_of_bcs.append(bc_coeff)

    #    L_moments = samlmu(np.delete(opinion_distr,influencers))
    #    L_skew.append(L_moments[0])
    #    L_kurt.append(L_moments[1])
    all_bc_values.append(list_of_bcs)
    mean_bc.append(np.mean(list_of_bcs))
    std_bc.append(np.std(list_of_bcs))
    quantiles_bc.append((np.quantile(list_of_bcs, 0.25),np.quantile(list_of_bcs, 0.5),np.quantile(list_of_bcs, 0.75)))
    peaks = peaks_of_distr(list_of_bcs,5/9,100)
    if np.isnan(peaks[0]):
        peak1_bc.append(0)
    else:
        peak1_bc.append(peaks[0])
    if np.isnan(peaks[1]):
        peak1_bc_alphas.append(0)
    else:
        peak1_bc_alphas.append(peaks[1])
    peak1_bc_mass.append(peaks[2])
    if np.isnan(peaks[3]):
        peak2_bc.append(0)
    else:
        peak2_bc.append(peaks[3])
    if np.isnan(peaks[4]):
        peak2_bc_alphas.append(0)
    else:
        peak2_bc_alphas.append(peaks[4])
    peak2_bc_mass.append(peaks[5])

mean_bc = np.array(mean_bc)
std_bc = np.array(std_bc)
peak1_bc = np.array(peak1_bc)
peak1_bc_alphas = np.array(peak1_bc_alphas)
peak1_bc_mass = np.array(peak1_bc_mass)
peak2_bc = np.array(peak2_bc)
peak2_bc_alphas = np.array(peak2_bc_alphas)
peak2_bc_mass = np.array(peak2_bc_mass)

#print(peak1_bc)
print(np.array([x_ax,peak1_bc_alphas,peak2_bc_alphas]).T)
print(np.array([x_ax,peak1_bc_mass,peak2_bc_mass]).T)
#print(peak2_bc)
#
# print(peak2_bc_alphas)

diff_mass = np.absolute(peak1_bc_mass-peak2_bc_mass)
critical_point = np.where(diff_mass==np.min(diff_mass))[0][0]

shade_upper_limit = np.array([np.min([1,mean_bc[i] + std_bc[i]]) for i in np.arange(len(mean_bc))])
shade_lower_limit = np.array([np.max([0,mean_bc[i] - std_bc[i]]) for i in np.arange(len(mean_bc))])

# ax = plt.gca()
# plt.plot(x_ax/100,mean_bc, color = '#ec7014')
# plt.fill_between(x_ax/100, shade_lower_limit, shade_upper_limit, alpha = 0.4, color = '#fec44f', linewidth=0)
# ax.axhline(5/9, linestyle='--', color='#404040')
# if mode == 'echo_chamber':
#     plt.ylabel(r'$BC_{\text{hom}}(\hat{b},\hat{b}_{NN})$')
# else:
#     plt.ylabel(r'$BC(\hat{b})$')
# plt.xlabel(r'fraction of stubborn users $s$')
# plt.ylim(0,1)
# plt.xlim(0,0.05)
# ax.axes.spines['top'].set_visible(False)
# ax.axes.spines['right'].set_visible(False)
# #plt.title('BC' + r'(b)$=$' + str(bimodality_index(distr)))
# if mode == 'echo_chamber':
#     plt.savefig(out_dir+'/echo_chambers_by_stubborn' + '_mean-value.pdf',bbox_inches='tight') # high-phi means phi=1.473
# else:
#     plt.savefig(out_dir+'/polarization_by_stubborn' + '_mean-value.pdf',bbox_inches='tight') # high-phi means phi=1.473
# plt.close()
# plt.clf()

## Plot 2 - median, 1st and 3rd quartiles
median_bc = np.array([el[1] for el in quantiles_bc]) # same as np.quantile(data[x],0.5)
shade_lower_limit = np.array([el[0] for el in quantiles_bc])
shade_upper_limit = np.array([el[2] for el in quantiles_bc])

ax = plt.gca()
plt.plot(x_ax/100,median_bc, color = '#ec7014')
plt.fill_between(x_ax/100, shade_lower_limit, shade_upper_limit, alpha = 0.4, color = '#fec44f', linewidth=0)
ax.axhline(5/9, linestyle='--', color='#404040')
if mode == 'echo_chamber':
    plt.ylabel(r'$BC_{\text{hom}}(\hat{b},\hat{b}_{NN})$')
else:
    plt.ylabel(r'$BC(\hat{b})$')
plt.xlabel(r'fraction of stubborn users $s$')
ax.axes.spines['top'].set_visible(False)
ax.axes.spines['right'].set_visible(False)
plt.ylim(0,1)
plt.xlim(0,0.05)
#plt.title('BC' + r'(b)$=$' + str(bimodality_index(distr)))
if mode == 'echo_chamber':
    plt.savefig(os.path.join(out_path, 'figS4a.pdf'), bbox_inches='tight')
else:
    plt.savefig(os.path.join(out_path,'/polarization_by_stubborn' + '_median.pdf'),bbox_inches='tight')
plt.close()
plt.clf()

# ## Plot 3 - ALL points

# ax = plt.gca()
# x_axis = np.array([])
# for x in x_ax:
#     x_axis = np.concatenate((x_axis,np.full(500,x/100)))
# y_axis = np.array([])
# for i in np.arange(len(x_ax)):
#     y_axis = np.concatenate((y_axis,np.array(all_bc_values[i])))
# plt.scatter(x_axis,y_axis, color = '#8856a7',alpha=0.05,s=20)
# ax.axhline(5/9, linestyle='--', color='#404040')
# if mode == 'echo_chamber':
#     plt.ylabel(r'$BC_{\text{hom}}(\hat{b},\hat{b}_{NN})$')
# else:
#     plt.ylabel(r'$BC(\hat{b})$')
# plt.xlabel(r'fraction of stubborn users $s$')
# ax.axes.spines['top'].set_visible(False)
# ax.axes.spines['right'].set_visible(False)
# plt.ylim(0,1)
# plt.xlim(0,0.05)
# #plt.title('BC' + r'(b)$=$' + str(bimodality_index(distr)))
# if mode == 'echo_chamber':
#     plt.savefig(out_dir+'/echo_chambers_by_stubborn' + '_all_vals.pdf',bbox_inches='tight')
# else:
#     plt.savefig(out_dir+'/polarization_by_stubborn' + '_all_vals.pdf',bbox_inches='tight')
# plt.close()
# plt.clf()

# ## Plot 4 - Two peaks, alpha for height

# ax = plt.gca()
# peak1_bc_alphas = np.array([min(2*a,1) for a in peak1_bc_alphas])
# #peak1_bc_alphas = np.array([1 if a>0 else 0 for a in peak1_bc_alphas])
# peak2_bc_alphas = np.array([min(2*a,1) for a in peak2_bc_alphas])
# #peak2_bc_alphas = np.array([1 if a>0 else 0 for a in peak2_bc_alphas])
# plt.scatter(x_ax/100,peak1_bc, color = '#f44336',alpha=peak1_bc_alphas, label='first peak', s=20)
# plt.scatter(x_ax/100,peak2_bc, color = '#3d85c6',alpha=peak2_bc_alphas, label='second peak', s=20)
# ax.axhline(5/9, linestyle='--', color='#404040')
# if mode == 'echo_chamber':
#     plt.ylabel(r'$BC_{\text{hom}}(\hat{b},\hat{b}_{NN})$')
# else:
#     plt.ylabel(r'$BC(\hat{b})$')
# plt.xlabel(r'fraction of stubborn users $s$')
# ax.axes.spines['top'].set_visible(False)
# ax.axes.spines['right'].set_visible(False)
# plt.ylim(0,1)
# plt.xlim(0,0.05)
# leg = plt.legend(loc='lower right',frameon=False)
# for lh in leg.legend_handles:
#     lh.set_alpha(np.ones(len(peak1_bc_alphas)))
# #plt.title('BC' + r'(b)$=$' + str(bimodality_index(distr)))
# if mode == 'echo_chamber':
#     plt.savefig(out_dir+'/echo_chambers_by_stubborn' + '_peaks_alpha-for-height.pdf',bbox_inches='tight')
# else:
#     plt.savefig(out_dir+'/polarization_by_stubborn' + '_peaks_alpha-for-height.pdf',bbox_inches='tight')
# plt.close()
# plt.clf()

## Plot 5 - Two peaks, alpha for mass

critical_y_min, _, _, critical_y_max, _, _ = peaks_of_distr(all_bc_values[critical_point],5/9,100)

ax = plt.gca()
plt.scatter(x_ax/100,peak1_bc, color = '#f44336',alpha=peak1_bc_mass, label='first peak', s=20)
plt.scatter(x_ax/100,peak2_bc, color = '#3d85c6',alpha=peak2_bc_mass, label='second peak', s=20)


#ax.axhline(5/9, linestyle='--', color='#404040')
ax.axvline(0.024, critical_y_min, critical_y_max, linestyle='--', color='#8856a7', alpha=0.5) # 
ax.text(0.021, 0.05, r'$s_1$', horizontalalignment='center') # s = 2.1
ax.text(0.0227, 0.5, r'$s_c$', horizontalalignment='center') # s = 2.4
ax.text(0.028, 0.92, r'$s_2$', horizontalalignment='center') # s = 2.8
if mode == 'echo_chamber':
    plt.ylabel(r'$BC_{\text{hom}}(\hat{b},\hat{b}_{NN})$')
else:
    plt.ylabel(r'$BC(\hat{b})$')
plt.xlabel(r'fraction of stubborn users $s$')
ax.axes.spines['top'].set_visible(False)
ax.axes.spines['right'].set_visible(False)
plt.ylim(0,1)
plt.xlim(0,0.05)
leg = plt.legend(loc='center right',frameon=False)
for lh in leg.legend_handles:
    lh.set_alpha(np.ones(len(peak1_bc_alphas)))
#plt.title('BC' + r'(b)$=$' + str(bimodality_index(distr)))
if mode == 'echo_chamber':
    plt.savefig(os.path.join(out_path, 'fig6a.pdf'), bbox_inches='tight')
else:
    plt.savefig(os.path.join(out_path, '/polarization_by_stubborn' + '_peaks_alpha-for-mass.pdf'),bbox_inches='tight') ##### SUBFIG-A
plt.close()
plt.clf()

# Plot 5 - Three BC distributions, for x = 1.5, 1.9, 2.4

ax = plt.gca()
x_axis = np.linspace(0,1,10000)

y1_distr = all_bc_values[21]
kde1 = gaussian_kde(y1_distr, bw_method=0.1)
y1_ax = kde1.evaluate(x_axis)

y2_distr = all_bc_values[24]
kde2 = gaussian_kde(y2_distr, bw_method=0.1)
y2_ax = kde2.evaluate(x_axis)

y3_distr = all_bc_values[28]
kde3 = gaussian_kde(y3_distr, bw_method=0.1)
y3_ax = kde3.evaluate(x_axis)

plt.plot(x_axis,y1_ax, color = '#c994c7', label='$s_1=0.021$')
plt.plot(x_axis,y2_ax, color = '#e7298a', label='$s_c=0.024$')
plt.plot(x_axis,y3_ax, color = '#91003f', label='$s_2=0.028$')
ax.axvline(5/9, 0, 0.4, linestyle='--', color='#404040')
#ax.text(5/9, 2, r'bimodality threshold', horizontalalignment='center', fontsize='small') # i = 1.8
plt.ylabel('density')
if mode == 'echo_chamber':
    plt.xlabel(r'$BC_{\text{hom}}(\hat{b},\hat{b}_{NN})$')
else:
    plt.xlabel(r'$BC(\hat{b})$')
ax.axes.spines['top'].set_visible(False)
ax.axes.spines['right'].set_visible(False)
# ax.axes.set(yticklabels=[])
# ax.axes.tick_params(left=False)
# ticks = [0,0.2,0.4,5/9,0.6,0.8,1.0]
# ax.set_xticks(ticks)
# dic = {5/9 : "5/9"}
# labels = [ticks[i] if t not in dic.keys() else dic[t] for i,t in enumerate(ticks)]
# ax.set_xticklabels(labels)
plt.xlim(0,1)
y_min, y_max = ax.axes.get_ylim()
y_min = -0.01
plt.ylim(y_min,y_max)
leg = plt.legend(loc='best',frameon=False)
for lh in leg.legend_handles:
    lh.set_alpha(1)
#plt.title('BC' + r'(b)$=$' + str(bimodality_index(distr)))
if mode == 'echo_chamber':
    plt.savefig(os.path.join(out_path, 'fig6b.pdf'), bbox_inches='tight')
else:
    plt.savefig(os.path.join(out_path,'/polarization-_by_stubborn' + '_peaks_three-samples.pdf'), bbox_inches='tight') ##### SUBFIG-B
plt.close()
plt.clf()













