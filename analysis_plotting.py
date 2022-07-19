import numpy as np
import statistics as stat
from scipy.stats import spearmanr

BEHAVIORS = ['attack', 'close_by', 'direct_competition', 'foraging_vs_exploration',
             'investigation', 'separate_exploration', 'separate_foraging', 'travel_away', 'travel_towards']
LABELDICT = {'attack': 'attack', 'close_by': 'close by', 'direct_competition': 'direct competition',
             'foraging_vs_exploration': 'foraging vs exploration', 'investigation': 'investigation',
             'separate_exploration': 'separate exploration', 'separate_foraging': 'separate foraging',
             'travel_away': 'travel away', 'travel_towards': 'travel towards'}
BAR_LABELS = ['attack', 'close\nby', 'direct\ncompetition', 'foraging vs\nexploration',
              'investigation', 'separate\nexploration', 'separate\nforaging', 'travel\naway', 'travel\ntowards']


## Behaviors repartition pie chart ####################################################################################


def behavior_total_durations(videos: str):
    total_durations = np.zeros(len(BEHAVIORS))
    for n in videos:
        behavior_vid_durations = []
        video_dict = np.load('data_video_' + n + '.npz', allow_pickle=True)
        behavior_data = video_dict['behavior_data']
        fr = video_dict['annot_fr']
        for behavior in BEHAVIORS:
            behavior_vid_durations.append(sum(behavior_data.item()[behavior]['duration'])/fr)
        total_durations += behavior_vid_durations
    return total_durations


def label_values(pct):
    if pct < 2.5:
        return None
    else:
        return str(int(pct)) + '%'


def label_time(durations):
    time_labels = []
    for time in durations:
        if time//3600 > 0:
            time_labels.append('> ' + str(int(time // 3600)) + ' h')
        elif time//60 > 0:
            time_labels.append('> ' + str(int(time // 60)) + ' mn')
        else:
            time_labels.append('> ' + str(int(time)) + ' s')
    return np.array([time_labels])


def plot_pie_chart_time_repartition(ax, durations, filename):
    behaviors_labels = np.array([LABELDICT[behavior] for behavior in BEHAVIORS])
    labels = np.char.add(behaviors_labels, np.array([" " for _ in range(len(BEHAVIORS))]))
    labels = np.char.add(labels, label_time(durations))
    ax.pie(durations, labels=labels[0],
           autopct=lambda pct: label_values(pct),
           shadow=False, startangle=0, counterclock=False, radius=1.3)
  #  ax.set_title(label='total duration per behavior, '+filename, y=-0.3)


def time_repart_subplot(ax, videos: str = '12345'):
    durations = behavior_total_durations(videos=videos)
    if videos == '12345':
        filename = 'all videos'
    else:
        filename = 'video '
        for n in videos[:-1]:
            filename += n + ', '
        filename += videos[-1]
    plot_pie_chart_time_repartition(ax, durations, filename)


## Duration variability ####################################################################################


def total_time_variability_hist(videos: str = '12345'):
    total_time = np.zeros([len(videos), len(BEHAVIORS)])
    for i, n in enumerate(videos):
        total_time[i] = np.array(behavior_total_durations(n))
    CV = np.std(total_time, axis=0)/np.mean(total_time, axis=0)
    return CV


def time_variability_hist_subplot(ax, videos: str = '12345'):
    CV = total_time_variability_hist(videos)
    ax.bar([k for k in range(9)], CV, tick_label=BAR_LABELS, width=0.6,
           color = 'lightblue', edgecolor = 'black', linewidth = 1)
    ax.set_ylabel(ylabel = 'total duration CV (mean/std) among videos')
   # ax.set_title(label = 'Total behavior duration CV among videos', y =0)


## Timeline #########################################################################################################


def create_timeline(n_video: str):
    video_dict = np.load('data_video_' + n_video + '.npz', allow_pickle=True)
    behavior_data = video_dict['behavior_data'].item()
    annot_framerate = video_dict['annot_fr']
    annot_start = video_dict['annot_start']
    annot_end = video_dict['annot_end']
    time = np.arange(start=annot_start, stop=annot_end, step=1/annot_framerate)
    timeline = np.empty(np.shape(time), dtype='<U30')
    for behavior in BEHAVIORS:
        for k in range(len(behavior_data[behavior]['start'])):
            behavior_start_frame = int(behavior_data[behavior]['start'][k])
            behavior_end_frame = int(behavior_data[behavior]['end'][k])
            timeline[behavior_start_frame: behavior_end_frame] = behavior
    return time, timeline


#### mean, std of duration and frequency ##############################################################################


def get_mean_and_std_duration(videos: str = '12345'):
    durations = {bhv: [] for bhv in BEHAVIORS}
    for n in videos:
        vid_dict = np.load('data_video_' + n + '.npz', allow_pickle=True)
        behavior_data = vid_dict['behavior_data'].item()
        fr = vid_dict['annot_fr']
        for i, bhv in enumerate(BEHAVIORS):
            durations[bhv] += list(np.array(behavior_data[bhv]['duration'])/fr)
    return [np.mean(durations[bhv]) for bhv in BEHAVIORS], [np.std(durations[bhv]) for bhv in BEHAVIORS]


def get_frequencies(videos: str = '12345'):
    total_time = 0
    frequencies = np.zeros(len(BEHAVIORS))
    for n in videos:
        video_dict = np.load('data_video_' + n + '.npz', allow_pickle=True)
        behavior_data = video_dict['behavior_data'].item()
        start = video_dict['annot_start']
        end = video_dict['annot_end']
        total_time += end-start
        frequencies += [len(behavior_data[bhv]['duration']) for bhv in BEHAVIORS]
    return 3600*frequencies/total_time


def subplot_mean(ax, videos: str = '12345'):
    mean, std = get_mean_and_std_duration(videos)
    yerr = np.array([[0, stdb] for stdb in std]).T
    ax.bar([k for k in range(len(BEHAVIORS))], mean, tick_label=BAR_LABELS, width=0.8)
    (_, caps, _) = ax.errorbar([k for k in range(len(BEHAVIORS))], mean, yerr = yerr, capsize = 10, ls = 'none',
                               color='black')
    for cap in caps:
        cap.set_markeredgewidth(1)
    # ax.set_title(label = 'mean duration (blue) and std (black) of behaviours', y=0)


def boxplot_durations(ax, videos: str = '12345'):
    durations = [[] for _ in BEHAVIORS]
    for n in videos:
        vid_dict = np.load('data_video_' + n + '.npz', allow_pickle=True)
        behavior_data = vid_dict['behavior_data'].item()
        fr = vid_dict['annot_fr']
        for i, bhv in enumerate(BEHAVIORS):
            durations[i] += list(np.array(behavior_data[bhv]['duration']) / fr)
    bp = ax.boxplot(durations, whis = (10,90), showfliers = False, showmeans = True, meanline = True, labels = BAR_LABELS,
               patch_artist= True, medianprops = {'linewidth': 1.5}, meanprops = {'linewidth': 1.5})
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_linewidth(1)
    ax.set_ylabel(ylabel = 'behaviour duration (s)')


def subplot_frequencies(ax, videos: str = '12345'):
    ax.bar([k for k in range(len(BEHAVIORS))], get_frequencies(videos), tick_label=BAR_LABELS, width=0.6,
           color = 'lightblue', edgecolor = 'black', linewidth = 1)
    ax.set_ylabel(ylabel = r'frequencies (h$^{-1}$)')


#### behavior distributions ###########################################################################################


def get_distribution_over_time(videos: str = '12345', timebin: int = 300, binstart: int = 0, binstop: int = 5000):
    distributions = {bhv: [] for bhv in BEHAVIORS}
    bins = np.arange(start=binstart, stop=binstop, step=timebin)
    for n in videos:
        video_dict = np.load('data_video_' + n + '.npz', allow_pickle=True)
        behavior_data = video_dict['behavior_data'].item()
        fr = video_dict['annot_fr']
        for bhv in BEHAVIORS:
            start_times = np.array(behavior_data[bhv]['start'])/fr
            distributions[bhv].append(np.histogram(start_times, bins=bins)/timebin)
    return {bhv: np.mean(distributions[bhv], axis=0) for bhv in BEHAVIORS}

"""
def get_threshold_change_triggered_distribution(videos: str = '12345', timebin=300, binstart=0, binstop=5000):
    distributions = {bhv: [] for bhv in BEHAVIORS}
    for n in videos:
        video_dict = np.load('data_video_' + n + '.npz', allow_pickle=True)
        time, timeline = create_timeline(n)
        fr = video_dict['annot_fr']
        threshold_change_time = video_dict['threshold_change'].item()['changetime']
        bins = np.arange(start=binstart-threshold_change_time, stop=binstop-threshold_change_time, step=timebin)
        for bhv in BEHAVIORS:
            bhv_occurences = time[timeline == bhv]
            distributions[bhv].append(np.histogram(bhv_occurences - threshold_change_time, bins=bins)/timebin)
    return bins, {bhv: np.mean(distributions[bhv], axis=0) for bhv in BEHAVIORS}
"""


def get_tctf(videos:str = '12345', timebin=500, binstart=0, binstop=5000):
    distributions = {bhv: np.zeros(int((binstop - binstart) / timebin) -1 ) for bhv in BEHAVIORS}
    for n in videos:
        video_dict = np.load('data_video_' + n + '.npz', allow_pickle=True)
        time, timeline = create_timeline(n)
        fr = video_dict['annot_fr']
        threshold_change_time = video_dict['threshold_change'].item()['changetime']
        bins = np.arange(start=binstart-threshold_change_time, stop=binstop-threshold_change_time, step=timebin)
        for bhv in BEHAVIORS:
            bhv_occurences = time[timeline == bhv] - threshold_change_time
            h, b = np.histogram(bhv_occurences, bins = bins)
            distributions[bhv] += h / (5 * timebin * fr)
    return b, distributions


def tctd_subplot(ax, videos: str = '12345'):
    bins, tctd = get_tctf(videos)
    for bhv in BEHAVIORS:
        ax.plot(bins[:-1], tctd[bhv], label=LABELDICT[bhv])
        ax.legend(loc = 'upper left')

#### tc curve ########################################################################################################

COMPETITION = ['direct_competition', 'close_by', 'travel_towards']
SOLO = ['separate_exploration', 'travel_away', 'separate_foraging', 'foraging_vs_exploration']
OTHER = ['attack', 'investigation']

def tc_behavioral_curve(ax, videos: str = '12345'):
    thresholds = []
    p = []
    for n in videos:
        video_dict = np.load('data_video_' + n + '.npz', allow_pickle=True)
        time, timeline = create_timeline(n)
        fr = video_dict['annot_fr']
        tc = video_dict['threshold_change'].item()
        tc_time = tc['changetime']
        t0 = tc['init_threshold']
        t1 = tc['new_threshold']
        solocount = np.sum(np.isin(timeline[time < tc_time], SOLO))
        compet_count = np.sum(np.isin(timeline[time < tc_time], COMPETITION))
        thresholds.append(t0)
        p.append(compet_count /solocount + compet_count)
        solocount = np.sum(np.isin(timeline[time > tc_time], SOLO))
        compet_count = np.sum(np.isin(timeline[time > tc_time], COMPETITION))
        thresholds.append(t1)
        p.append(compet_count / solocount + compet_count)
    r = spearmanr(thresholds, p)[0]
    ax.scatter(thresholds, p)
    m, b = np.polyfit(thresholds, p, 1)
    ax.plot(thresholds, m*np.array(thresholds)+b, color='red', label='r = ' + str(r))
    ax.legend()

#### markov analysis #################################################################################################

def get_next_behaviors_prob(bhv, timeline):
    other_behaviors = BEHAVIORS.copy()
    other_behaviors.remove(bhv)
    next_behaviors_count = {b: 0 for b in other_behaviors}
    i: int = 0
    while i < np.shape(timeline)[0]:
        if timeline[i] == bhv:
            while i + 1 < np.shape(timeline)[0] and (timeline[i] == bhv or timeline[i] == ''):
                i += 1
            if timeline[i] != bhv and timeline[i] != '':
                next_behaviors_count[timeline[i]] += 1
        else:
            i += 1
    for b in other_behaviors:
        next_behaviors_count[b] /= np.sum([next_behaviors_count[beh] for beh in other_behaviors])
    return next_behaviors_count


def get_previous_behavior_prob(bhv, timeline):
    return get_next_behaviors_prob(bhv, np.flip(timeline))


def next_behavior_pie_subplot(ax, bhv, videos='12345'):
    nbp = np.zeros(len(BEHAVIORS) - 1)
    for n in videos:
        timeline = create_timeline(n)
        nbp += get_next_behaviors_prob(bhv, timeline)
    ax.pie(nbp,
           autopct=lambda pct: label_values(pct),
           shadow=True)


def prev_behavior_pie_subplot(ax, bhv, videos='12345'):
    nbp = np.zeros(len(BEHAVIORS) - 1)
    for n in videos:
        timeline = create_timeline(n)
        nbp += get_previous_behavior_prob(bhv, timeline)
    ax.pie(nbp,
           autopct=lambda pct: label_values(pct),
           shadow=True)


'''
def build_markov_chain_matrix_v0():
    behaviors = ['attack', 'close_by', 'direct_competition', 'foraging_vs_exploration',
                       'investigation', 'separate_exploration', 'separate_foraging', 'travel_away', 'travel_towards']
    next_behaviour_probs = {behavior: characterize_behavior(behavior)['next_behaviour_prob'] for behavior in behaviors}
    P = np.zeros([len(behaviors), len(behaviors)])
    for i, behavior_1 in enumerate(behaviors):
        for j, behavior_2 in enumerate(behaviors):
            if i == j:
                pass
            else:
                P[i, j] = next_behaviour_probs[behavior_1][behavior_2]
    return P
'''
