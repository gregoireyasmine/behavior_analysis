import numpy as np


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
        behavior_data = np.load('data_video_' + n + '.npz')['behavior_data']
        for behavior in BEHAVIORS:
            behavior_vid_durations += sum(behavior_data[behavior]['duration'])
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
    ax.pie(behavior_total_durations, labels=labels[0],
           autopct=lambda pct: label_values(pct),
           shadow=True, startangle=0, counterclock=False, radius=1.8)
    ax.set_title(label='total duration per behavior, '+filename, y=-0.5)


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
    std = np.std(total_time, axis=0)/np.mean(total_time, axis=0)
    return std


def time_variability_hist_subplot(ax, videos: str = '12345'):
    std = total_time_variability_hist(videos)
    ax.bar([1.5*k for k in range(9)], std, tick_label=BAR_LABELS, width=0.4)


## Timeline #########################################################################################################


def create_timeline(n_video: str):

    video_dict = np.load('data_video_' + n_video + '.npz')
    behavior_data = video_dict['behavior_data']
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
    durations = [[]] * len(BEHAVIORS)
    for n in videos:
        behavior_data, fr = np.load('data_video_' + n + '.npz')['behavior_data', 'annot_fr']
        for i, bhv in enumerate(BEHAVIORS):
            durations[i] += list(np.array(behavior_data[bhv]['duration'])/fr)
    return [np.mean(durations[i]) for i in range(len(BEHAVIORS))], [np.std(durations[i]) for i in range(len(BEHAVIORS))]


def get_frequencies(videos: str = '12345'):
    total_time = 0
    frequencies = np.zeros(len(BEHAVIORS))
    for n in videos:
        behavior_data, start, end = np.load('data_video_' + n + '.npz')['behavior_data', 'annot_start', 'annot_end']
        total_time += start-end
        frequencies += len(behavior_data['duration'])
    return frequencies/total_time


def subplot_mean(ax, videos: str = '12345'):
    mean, std = get_mean_and_std_duration(videos)
    ax.bar([k for k in range(len(BEHAVIORS))], mean, tick_label=BAR_LABELS, width=0.8, yerr=std)


def subplot_frequencies(ax, videos: str = '12345'):
    ax.bar([k for k in range(len(BEHAVIORS))], get_frequencies(videos), tick_label=BAR_LABELS, width=0.8)


#### behavior distributions ###########################################################################################


def get_distribution_over_time(videos: str = '12345', timebin: int = 300, binstart: int = 0, binstop: int = 5000):
    distributions = {bhv: [] for bhv in BEHAVIORS}
    for n in videos:
        behavior_data, fr = np.load('data_video_' + n + '.npz')['behavior_data', 'annot_fr']
        bins = np.arange(start=binstart, stop=binstop, step=timebin)
        for bhv in BEHAVIORS:
            start_times = np.array(behavior_data[bhv]['start'])/fr
            distributions[bhv].append(np.histogram(start_times, bins=bins)/timebin)
    return {bhv: np.mean(distributions[bhv], axis=0) for bhv in BEHAVIORS}


def get_threshold_change_triggered_distribution(videos: str = '12345', timebin=300, binstart=0, binstop=5000):
    distributions = {bhv: [] for bhv in BEHAVIORS}
    for n in videos:
        behavior_data, fr, threshold_change = np.load('data_video_' + n + '.npz')['threshold_change']
        threshold_change_time = threshold_change['changetime']
        bins = np.arange(start=binstart-threshold_change_time, stop=binstop+threshold_change_time, step=timebin)
        for bhv in BEHAVIORS:
            start_times = np.array(behavior_data[bhv]['start']) / fr
            distributions[bhv].append(np.histogram(start_times - threshold_change_time, bins=bins)/timebin)
    return bins, {bhv: np.mean(distributions[bhv], axis=0) for bhv in BEHAVIORS}


def tctd_subplot(ax, videos: str = '12345'):
    bins, tctd = get_threshold_change_triggered_distribution(videos)
    for bhv in BEHAVIORS:
        ax.plot(bins, tctd, label=LABELDICT[bhv])
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
