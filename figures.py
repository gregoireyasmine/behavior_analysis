import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
import matplotlib.image as mpimg
from subprocess import check_call
from analysis_plotting import *
from markovchain import MarkovChain

mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 13
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['text.color'] = '#000000'
mpl.rcParams['axes.titlecolor'] = '#000000'
mpl.rcParams['xtick.color'] = '#000000'
mpl.rcParams['ytick.color'] = '#000000'
mpl.rcParams['xtick.labelcolor'] = '#000000'
mpl.rcParams['ytick.labelcolor'] = '#000000'
mpl.rcParams['axes.labelcolor'] = '#000000'
mpl.rcParams['axes.titleweight'] = 'bold'


###  FIG 1 : basic data analysis

fig, ax = plt.subplot_mosaic([['total_durations', 'variability'], ['mean_duration', 'frequency']], figsize=(20, 11))
BEHAVIORS = ['attack', 'close_by', 'direct_competition', 'foraging_vs_exploration',
             'investigation', 'separate_exploration', 'separate_foraging', 'travel_away', 'travel_towards']
time_repart_subplot(ax['total_durations'])
time_variability_hist_subplot(ax['variability'])
boxplot_durations(ax['mean_duration'])
subplot_frequencies(ax['frequency'])
titles = ['A', 'B', 'C', 'D']
xs = [-0.6, -0.05, -0.05, -0.05]
for i, axx in enumerate(ax):
    ax[axx].set_title(label = titles[i], loc = 'left', y=1.05, x=xs[i])
plt.tight_layout()
plt.savefig('basic_data_analysis')

### FIG 2 : behaviour evolution (tc)

fig, ax = plt.subplot_mosaic([['tc_triggered', 'behavioral_curve']], figsize = (13, 8))
tctd_subplot(ax['tc_triggered'])
tc_behavioral_curve(ax['behavioral_curve'])
plt.tight_layout()
plt.savefig('tc analysis')


### FIG3 : plotting some markov chains




""""
def plot_characterization(behavior):
    behaviors = ['attack', 'close_by', 'direct_competition', 'foraging_vs_exploration',
                 'investigation', 'separate_exploration', 'separate_foraging', 'travel_away', 'travel_towards']
    other_behaviors = behaviors.copy()
    other_behaviors.remove(behavior)
    fig, ax = plt.subplot_mosaic([['duration', 'next', 'prev'],
                                  ['frequency', 't_distrib', 't_distrib']],
                                 figsize=(20, 12), gridspec_kw=dict(width_ratios=[0.3, 1, 1], height_ratios=[1, 1]))
    fig.patch.set_facecolor('#232323')
    for axx in ax.keys():
        ax[axx].patch.set_facecolor('#ababab')
    characterization = characterize_behavior(behavior)
    ax['duration'].bar([1], [characterization['mean_duration']], tick_label=' ', width=0.2)
    ax['duration'].set_title(label='mean duration (s)')
    ax['frequency'].bar([1], [characterization['frequency']*3600], tick_label=' ', width=0.2)
    ax['frequency'].set_title(label='frequency (1/h)')
    h, bins = characterization['t_distrib']
    ax['t_distrib'].bar(bins[:-1], h, width=bins[1]-bins[0])
    ax['t_distrib'].set_title(label='distribution of behavior over time')
    ax['t_distrib'].plot([0, 0], [0, 1.2 * max(h)], color = 'red')
    _, _, autotexts = ax['next'].pie([characterization['next_behaviour_prob'][b] for b in other_behaviors],
                   autopct=lambda pct: label_values(pct),
                   shadow=True)
    for autotext in autotexts:
        autotext.set_color('white')
    ax['next'].set_title('next behavior distribution')

    _, _, autotexts = ax['prev'].pie([characterization['prev_behaviour_prob'][b] for b in other_behaviors],
                   autopct=lambda pct: label_values(pct),
                   shadow=True)
    for autotext in autotexts:
        autotext.set_color('white')
    ax['prev'].legend(labels=other_behaviors, loc='upper right', bbox_to_anchor=(1, 0, 0.5, 1))
    ax['prev'].set_title('previous behavior distribution')
    fig.suptitle(behavior + ' behavior characterization')
    fig.savefig(behavior + '_characterization')
    plt.close()


def plot_markov_chain_v0():
    behaviors = ['attack', 'close\nby', 'direct\ncompetition', 'foraging vs\nexploration',
                        'investigation', 'separate\nexploration', 'separate\nforaging', 'travel\naway',
                 'travel\ntowards']
    poses = [(0, 0), (1, 4), (0, 4), (2, 1), (2, 0), (0, 2), (2, 2), (2, 3), (0, 3)]

    P = build_markov_chain_matrix_v0()
    G = nx.MultiDiGraph()
    labels = {}
    edge_labels = {}
    for i, origin_behavior in enumerate(behaviors):
        for j, destination_behavior in enumerate(behaviors):
            rate = P[i][j]
            if rate > 0.1:
                G.add_edge(origin_behavior, destination_behavior, weight=rate, label="{:.02f}".format(rate))
                edge_labels[(origin_behavior, destination_behavior)] = label = "{:.02f}".format(rate)
    node_size = 200
    pos = {behavior: list(poses[i]) for i, behavior in enumerate(behaviors)}
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight=2)
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    plt.axis('off')
    plt.savefig('markov_chain_v0')
"""
'''
behaviors = ['attack', 'close_by', 'direct_competition', 'foraging_vs_exploration',
             'investigation', 'separate_exploration', 'separate_foraging', 'travel_away', 'travel_towards']
for behavior in behaviors:
    plot_characterization(behavior)

'''

