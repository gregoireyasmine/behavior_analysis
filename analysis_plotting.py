import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import curve_fit
from markovchain import MarkovChain
import matplotlib.pyplot as plt
from math import sqrt, pi
from scipy.stats import norm, wilcoxon

VERBOSE = 1

BEHAVIORS = ['attack', 'close_by', 'direct_competition', 'foraging_vs_exploration',
             'investigation', 'separate_exploration', 'separate_foraging', 'travel_away', 'travel_towards']

LABELDICT = {'attack': 'attack', 'close_by': 'close by', 'direct_competition': 'direct competition',
             'foraging_vs_exploration': 'foraging vs exploration', 'investigation': 'investigation',
             'separate_exploration': 'separate exploration', 'separate_foraging': 'separate foraging',
             'travel_away': 'travel away', 'travel_towards': 'travel towards'}

BAR_LABELS = ['attack', 'close\nby', 'direct\ncomp.', 'forag.\n vs\nexplor.',
              'investig.', 'sep.\nexplor.', 'sep.\nforag.', 'travel\naway', 'travel\ntowards']


## Behaviors repartition pie chart ####################################################################################


def behavior_total_durations(videos: str):
    total_durations = np.zeros(len(BEHAVIORS))
    for n in videos:
        behavior_vid_durations = []
        video_dict = np.load('data_video_' + n + '.npz', allow_pickle=True)
        behavior_data = video_dict['behavior_data']
        fr = video_dict['annot_fr']
        for behavior in BEHAVIORS:
            behavior_vid_durations.append(sum(behavior_data.item()[behavior]['duration']) / fr)
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
        if time // 3600 > 0:
            time_labels.append('> ' + str(int(time // 3600)) + ' h')
        elif time // 60 > 0:
            time_labels.append('> ' + str(int(time // 60)) + ' mn')
        else:
            time_labels.append('> ' + str(int(time)) + ' s')
    return np.array([time_labels])


def plot_pie_chart_time_repartition(ax, durations):
    behaviors_labels = np.array([LABELDICT[behavior] for behavior in BEHAVIORS])
    labels = np.char.add(behaviors_labels, np.array([" " for _ in range(len(BEHAVIORS))]))
    labels = np.char.add(labels, label_time(durations))
    ax.pie(durations, labels=labels[0],
           autopct=lambda pct: label_values(pct),
           shadow=False, startangle=0, counterclock=False, radius=1)


def time_repart_subplot(ax, videos: str = '12345'):
    durations = behavior_total_durations(videos=videos)
    if videos == '12345':
        filename = 'all videos'
    else:
        filename = 'video '
        for n in videos[:-1]:
            filename += n + ', '
        filename += videos[-1]
    plot_pie_chart_time_repartition(ax, durations)
    ax.set_title(label=filename)


## Duration variability ####################################################################################


def total_time_variability_hist(videos: str = '12345'):
    total_time = np.zeros([len(videos), len(BEHAVIORS)])
    for i, n in enumerate(videos):
        total_time[i] = np.array(behavior_total_durations(n))
    cv = np.std(total_time, axis=0) / np.mean(total_time, axis=0)
    return cv


def time_variability_hist_subplot(ax, videos: str = '12345'):
    cv = total_time_variability_hist(videos)
    ax.bar([k for k in range(9)], cv, tick_label=BAR_LABELS, width=0.6,
           color='lightblue', edgecolor='black', linewidth=1)
    ax.set_ylabel(ylabel='total duration CV (mean/std) among videos')
    # ax.set_title(label = 'Total behavior duration CV among videos', y =0)


## Timeline #########################################################################################################


def create_timeline(n_video: str):
    video_dict = np.load('data_video_' + n_video + '.npz', allow_pickle=True)
    behavior_data = video_dict['behavior_data'].item()
    annot_framerate = video_dict['annot_fr']
    annot_start = video_dict['annot_start']
    annot_end = video_dict['annot_end']
    time = np.arange(start=annot_start, stop=annot_end, step=1 / annot_framerate)
    timeline = np.empty(np.shape(time), dtype='<U30')
    for behavior in BEHAVIORS:
        for k in range(len(behavior_data[behavior]['start'])):
            behavior_start_frame = int(behavior_data[behavior]['start'][k])
            behavior_end_frame = int(behavior_data[behavior]['end'][k])
            timeline[behavior_start_frame: behavior_end_frame] = behavior
    return time, timeline


def create_timeline_v2(n_video: str):
    video_dict = np.load('data_video_' + n_video + '.npz', allow_pickle=True)
    behavior_data = video_dict['behavior_data'].item()
    annot_framerate = video_dict['annot_fr']
    annot_start = video_dict['annot_start']
    annot_end = video_dict['annot_end']
    time = np.arange(start=annot_start, stop=annot_end, step=1 / annot_framerate)
    timeline = np.empty(np.shape(time), dtype=tuple)
    for behavior in BEHAVIORS:
        for k in range(len(behavior_data[behavior]['start'])):
            behavior_start_frame = int(behavior_data[behavior]['start'][k])
            behavior_end_frame = int(behavior_data[behavior]['end'][k])
            if len(behavior_data[behavior].keys()) > 3:
                individual_close_from_patch = behavior_data[behavior]['individual_close_from_patch'][k]
                patch = behavior_data[behavior]['patch_of_interest'][k]
            else:
                individual_close_from_patch = None
                patch = None
            timeline[behavior_start_frame: behavior_end_frame].fill((behavior, individual_close_from_patch, patch))
    return time, timeline


# mean, std of duration and frequency ##############################################################################

def get_mean_and_std_duration(videos: str = '12345'):
    durations = {bhv: [] for bhv in BEHAVIORS}
    for n in videos:
        vid_dict = np.load('data_video_' + n + '.npz', allow_pickle=True)
        behavior_data = vid_dict['behavior_data'].item()
        fr = vid_dict['annot_fr']
        for i, bhv in enumerate(BEHAVIORS):
            durations[bhv] += list(np.array(behavior_data[bhv]['duration']) / fr)
    return [np.mean(durations[bhv]) for bhv in BEHAVIORS], [np.std(durations[bhv]) for bhv in BEHAVIORS]


def get_frequencies(videos: str = '12345'):
    total_time = 0
    frequencies = np.zeros(len(BEHAVIORS))
    for n in videos:
        video_dict = np.load('data_video_' + n + '.npz', allow_pickle=True)
        behavior_data = video_dict['behavior_data'].item()
        start = video_dict['annot_start']
        end = video_dict['annot_end']
        total_time += end - start
        frequencies += [len(behavior_data[bhv]['duration']) for bhv in BEHAVIORS]
    return 3600 * frequencies / total_time


def subplot_mean(ax, videos: str = '12345'):
    mean, std = get_mean_and_std_duration(videos)
    yerr = np.array([[0, stdb] for stdb in std]).T
    ax.bar([k for k in range(len(BEHAVIORS))], mean, tick_label=BAR_LABELS, width=0.8)
    (_, caps, _) = ax.errorbar([k for k in range(len(BEHAVIORS))], mean, yerr=yerr, capsize=10, ls='none',
                               color='black')
    for cap in caps:
        cap.set_markeredgewidth(1)


def boxplot_durations(ax, videos: str = '12345'):
    durations = [[] for _ in BEHAVIORS]
    for n in videos:
        vid_dict = np.load('data_video_' + n + '.npz', allow_pickle=True)
        behavior_data = vid_dict['behavior_data'].item()
        fr = vid_dict['annot_fr']
        for i, bhv in enumerate(BEHAVIORS):
            durations[i] += list(np.array(behavior_data[bhv]['duration']) / fr)
    bp = ax.boxplot(durations, whis=(10, 90), showfliers=False, showmeans=True, meanline=True, labels=BAR_LABELS,
                    patch_artist=True, medianprops={'linewidth': 1.5}, meanprops={'linewidth': 1.5})
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_linewidth(1)
    ax.set_ylabel(ylabel='behaviour duration (s)')


def subplot_frequencies(ax, videos: str = '12345'):
    ax.bar([k for k in range(len(BEHAVIORS))], get_frequencies(videos), tick_label=BAR_LABELS, width=0.6,
           color='lightblue', edgecolor='black', linewidth=1)
    ax.set_ylabel(ylabel=r'frequencies (h$^{-1}$)')


#### behavior distributions ###########################################################################################


def get_distribution_over_time(videos: str = '12345', timebin: int = 300, binstart: int = 0, binstop: int = 5000):
    distributions = {bhv: [] for bhv in BEHAVIORS}
    bins = np.arange(start=binstart, stop=binstop, step=timebin)
    for n in videos:
        video_dict = np.load('data_video_' + n + '.npz', allow_pickle=True)
        behavior_data = video_dict['behavior_data'].item()
        fr = video_dict['annot_fr']
        for bhv in BEHAVIORS:
            start_times = np.array(behavior_data[bhv]['start']) / fr
            distributions[bhv].append(np.histogram(start_times, bins=bins) / timebin)
    return {bhv: np.mean(distributions[bhv], axis=0) for bhv in BEHAVIORS}


def get_tctf(videos: str = '12345', timebin=500, binstart=0,
             binstop=5000):  # local frequency variations triggered by threshold change
    distributions = {bhv: np.zeros(int((binstop - binstart) / timebin) - 1) for bhv in BEHAVIORS}
    for n in videos:
        video_dict = np.load('data_video_' + n + '.npz', allow_pickle=True)
        time, timeline = create_timeline(n)
        fr = video_dict['annot_fr']
        threshold_change_time = video_dict['threshold_change'].item()['changetime']
        bins = np.arange(start=binstart - threshold_change_time, stop=binstop - threshold_change_time, step=timebin)
        for bhv in BEHAVIORS:
            bhv_occurences = time[timeline == bhv] - threshold_change_time
            h, b = np.histogram(bhv_occurences, bins=bins)
            distributions[bhv] += h / (5 * timebin * fr)
    return b, distributions


def tctd_subplot(ax, videos: str = '12345'):
    bins, tctd = get_tctf(videos)
    for bhv in BEHAVIORS:
        ax.plot(bins[:-1], tctd[bhv], label=LABELDICT[bhv])
        ax.set_ylabel('behavior probability')
        ax.set_xlabel('time to threshold change (s)')
        ax.legend(loc='upper left')


#### tc curve ########################################################################################################

COMPETITION = ['direct_competition', 'close_by', 'travel_towards', 'attack']
SOLO = ['separate_exploration', 'travel_away', 'separate_foraging', 'foraging_vs_exploration']
OTHER = ['investigation']


def tc_behavioral_curve(ax, videos: str = '12345'):
    thresholds = []
    p = []
    for n in videos:
        video_dict = np.load('data_video_' + n + '.npz', allow_pickle=True)
        time, timeline = create_timeline(n)
        tc = video_dict['threshold_change'].item()
        tc_time = tc['changetime']
        t0 = tc['init_threshold']
        t1 = tc['new_threshold']
        solocount = np.sum(np.isin(timeline[time < tc_time], SOLO))
        compet_count = np.sum(np.isin(timeline[time < tc_time], COMPETITION))
        thresholds.append(t0)
        p.append(compet_count / (solocount + compet_count))
        solocount = np.sum(np.isin(timeline[time > tc_time], SOLO))
        compet_count = np.sum(np.isin(timeline[time > tc_time], COMPETITION))
        thresholds.append(t1)
        p.append(compet_count / (solocount + compet_count))
    r, p_value = spearmanr(thresholds, p)
    ax.scatter(thresholds, p)
    m, b = np.polyfit(thresholds, p, 1)
    ax.plot(thresholds, m * np.array(thresholds) + b, color='red', label='r = ' + str(r) + '    ' + 'p = '
                                                                         + str(p_value))
    ax.set_xlabel(xlabel='wheel threshold')
    ax.set_ylabel(ylabel='proportion of competitive behaviours')
    ax.legend(loc='upper left')


def tc_delta_bhv(ax, videos: str = '12345'):
    deltas = {bhv: {'pre': [], 'post': []} for bhv in BEHAVIORS}
    total_before_tc = 0
    total_after_tc = 0
    for n in videos:
        video_dict = np.load('data_video_' + n + '.npz', allow_pickle=True)
        time, timeline = create_timeline(n)
        tc = video_dict['threshold_change'].item()
        tc_time = tc['changetime']
        total_before_tc += np.sum(np.isin(timeline[time < tc_time], BEHAVIORS))
        total_after_tc += np.sum(np.isin(timeline[time > tc_time], BEHAVIORS))
        for bhv in BEHAVIORS:
            before_tc_count = np.sum(timeline[time < tc_time] == bhv)
            after_tc_count = np.sum(timeline[time > tc_time] == bhv)
            deltas[bhv]['pre'].append(before_tc_count / total_before_tc)
            deltas[bhv]['post'].append(after_tc_count / total_after_tc)
    p_val = {}
    for bhv in BEHAVIORS:
        stat, p = wilcoxon(deltas[bhv]['pre'], deltas[bhv]['post'])
        p_val[bhv] = p
    delta_means = {bhv: np.mean(np.array(deltas[bhv]['post']) - np.array(deltas[bhv]['pre'])) for bhv in BEHAVIORS}
    x_bars = [k for k in range(len(BEHAVIORS))]
    width = 0.6
    y_bars = [delta_means[bhv] for bhv in BEHAVIORS]
    colorid = [int((p_val[bhv] < 0.1) + (p_val[bhv] < 0.05)) for bhv in BEHAVIORS]
    color = [['lightyellow', 'orange', 'coral'][cid] for cid in colorid]
    ax.bar(x_bars, y_bars, tick_label=BAR_LABELS, width=0.6,
           color=color, edgecolor='black', linewidth=1)
    ax.set_ylabel('mean variation of proportion')
    return p_val


#### markov analysis #################################################################################################


def other_behaviors(b):
    others = BEHAVIORS.copy()
    others.remove(b)
    return others


def get_next_behaviors(bhv, timeline):
    next_behaviors_count = {b: 0 for b in BEHAVIORS}
    i: int = 0
    while i < np.shape(timeline)[0]:
        if timeline[i] == bhv:
            while i + 1 < np.shape(timeline)[0] and (timeline[i] == bhv or timeline[i] == ''):
                i += 1
            if timeline[i] != bhv and timeline[i] != '':
                next_behaviors_count[timeline[i]] += 1
        else:
            i += 1
    return next_behaviors_count


def get_previous_behaviors(bhv, timeline):
    return get_next_behaviors(bhv, np.flip(timeline))


def get_next_behaviors_same_included(bhv, timeline):
    next_behaviors_count = {b: 0 for b in BEHAVIORS}
    i: int = 0
    while i < np.shape(timeline)[0]:
        if timeline[i] == bhv and i + 1 < np.shape(timeline)[0]:
            i += 1
            while timeline[i] == '' and i + 1 < np.shape(timeline)[0]:
                i += 1
            if timeline[i] != '':
                next_behaviors_count[timeline[i]] += 1
        else:
            i += 1
    return next_behaviors_count


def next_behavior_pie_subplot(ax, bhv, videos='12345'):
    nbp = np.zeros(len(BEHAVIORS) - 1)
    for n in videos:
        timeline = create_timeline(n)
        nbp += get_next_behaviors(bhv, timeline)
    ax.pie(nbp,
           autopct=lambda pct: label_values(pct),
           shadow=True)


def prev_behavior_pie_subplot(ax, bhv, videos='12345'):
    nbp = np.zeros(len(BEHAVIORS) - 1)
    for n in videos:
        timeline = create_timeline(n)
        nbp += get_previous_behaviors(bhv, timeline)
    ax.pie(nbp,
           autopct=lambda pct: label_values(pct),
           shadow=True)


def get_markov_matrix(videos='12345', include_same=False):
    mm = {bhv: {bh: 0 for bh in BEHAVIORS} for bhv in BEHAVIORS}
    nbcounter = [get_next_behaviors, get_next_behaviors_same_included][include_same]
    for n in videos:
        _, timeline = create_timeline(n)
        for bhv in BEHAVIORS:
            nbc = nbcounter(bhv, timeline)
            for bh in nbc.keys():
                mm[bhv][bh] += nbc[bh]
    for bhv in BEHAVIORS:
        total = sum([mm[bhv][b] for b in mm[bhv].keys()])
        for bh in mm[bhv].keys():
            mm[bhv][bh] /= total
    return mm


def get_transition_probability(mm, state_1, state_2, steps,
                               forbidden_steps=None):
    """ Probability to go from one state to the other through [steps] intermediary steps"""
    if forbidden_steps is None:
        forbidden_steps = ['separate_foraging', 'separate_exploration',
                           'foraging_vs_exploration', 'direct_competition',
                           'close_by']
    p = 0
    for behavior in BEHAVIORS:
        #  for each other behavior
        if not (behavior in forbidden_steps):
            p += mm[state_1][behavior]
    p_i = 0
    for behavior_1 in BEHAVIORS:
        for behavior_2 in BEHAVIORS:
            if not ((behavior_1 in forbidden_steps) or (behavior_2 in forbidden_steps)):
                if behavior_1 != behavior_2:
                    p_i += mm[behavior_1][behavior_2]
    p *= p_i ** (steps - 1)
    p_i = 0
    for behavior in BEHAVIORS:
        if not (behavior in forbidden_steps):
            p_i += mm[behavior][state_2]
    p *= p_i
    return p


def collapse_markov(mm, states_dict: dict, videos='12345'):
    """ Collapse markov matrix to states of interest. """
    collapsed_mm = np.zeros((len(states_dict.keys()), len(states_dict.keys())))
    total_durations = behavior_total_durations(videos)
    for i, s1 in enumerate(states_dict.keys()):
        for j, s2 in enumerate(states_dict.keys()):
            s1_repart = total_durations[np.isin(BEHAVIORS, states_dict[s1])]
            s1_repart /= sum(s1_repart)
            p_transition = 0
            for k, b1 in enumerate(states_dict[s1]):
                for b2 in states_dict[s2]:
                    p_transition += mm[b1][b2] * s1_repart[k]
            collapsed_mm[i, j] = p_transition
    collapsed_mm = np.linalg.matrix_power(collapsed_mm, 30)
    collapsed_mm = np.around(collapsed_mm, decimals=2)
    return collapsed_mm


def markov_1(ax, videos='12345'):
    states = {'social_foraging': ['close_by', 'direct_competition'], 'travel_towards': ['travel_towards'],
              'travel_away': ['travel_away'],
              'notsocialforaging': ['attack', 'foraging_vs_exploration', 'investigation', 'separate_exploration',
                                    'separate_foraging']}  # ALPHABETIC ORDER FOR BHV
    complete_mm = get_markov_matrix(videos, include_same=True)
    collapsed_mm = collapse_markov(complete_mm, states_dict=states, videos=videos)
    mc = MarkovChain(collapsed_mm,
                     ['social\nforaging', 'travel\ntowards', 'travel\naway', 'non social\nor out of patch'])
    mc.draw(ax=ax, show=False)


def compute_angular_speeds(video: str):
    video_dict = np.load('data_video_' + video + '.npz', allow_pickle=True)
    wheel_data = video_dict['wheel_data'].item()
    angular_speed_df = pd.DataFrame()
    for side in 'right', 'left':
        if len(angular_speed_df) == 0:
            angular_speed_df['Time'] = wheel_data[side]['Time']
        delta_theta = wheel_data[side].diff()
        angular_speed_df[side] = delta_theta['angle'] / delta_theta['Time']
    return angular_speed_df


def characterize_angular_speed(videos='12345'):
    fig, ax = plt.subplots(4, 1)
    angular_speeds = np.array([])
    for n in videos:
        speed = compute_angular_speeds(n)
        for side in ['left', 'right']:
            angular_speeds = np.concatenate((angular_speeds, (speed[side][1:])))
    angular_speeds = angular_speeds[np.isfinite(angular_speeds)]

    def gaussian_func(x, m, sigma, c):
        return 1 / (sigma * sqrt(2 * pi)) * np.exp(-(x - m) ** 2 / (2 * sigma ** 2)) + c

    bin = 51

    angular_speeds = angular_speeds[np.isfinite(angular_speeds)]
    x0 = np.linspace(np.min(angular_speeds), np.max(angular_speeds), bin)
    y0, x0 = np.histogram(angular_speeds, bins=x0, density=True)

    ax[0].step(x0[:-1], y0)
    ax[0].grid()

    x_fit = x0[x0 < 0]
    y_fit = y0[x0[:-1] < 0]

    initial_guess = [0, 2, 0]
    popt, pcov = curve_fit(gaussian_func, x_fit, y_fit, p0=initial_guess)
    print(popt)

    xplot = np.linspace(np.min(angular_speeds), np.max(angular_speeds), bin)
    yplot = gaussian_func(xplot, *popt)
    ax[0].plot(xplot, yplot, ls='--', lw=1.2)

    ax[1].plot(xplot, 1 - norm.cdf(xplot, loc=popt[0], scale=popt[1]) + popt[2])

    ydiff = - yplot[:-1] + y0
    ax[2].plot(x0[:-1], ydiff)
    ax[2].grid()

    speed = compute_angular_speeds('1')
    ax[3].plot(speed['Time'][:50000], speed['left'][:50000], lw=0.4)
    ax[3].plot([0, speed['Time'][50000]], [12, 12], lw=0.4)

    fig.show()
    fig.savefig('angular_speeds.png')
    # an angular speed threshold of 12 seems good to define wheel activation


def behavior_vs_wheel_activation(time, timeline_v2, angular_speeds, threshold=12):
    tracking_data_completeness = {bhv: 0 for bhv in
                                  ['foraging_vs_exploration', 'other_patch_related',
                                   'separate_foraging', 'other_non_patch_related']}
    tracking_data_completeness['total'] = 0
    rw_activation = angular_speeds['right'] > threshold
    lw_activation = angular_speeds['left'] > threshold
    act_dict = {1: rw_activation, 2: lw_activation}
    angular_speeds_bin = angular_speeds['Time'][1] - angular_speeds['Time'][0]
    timeline_bin = time[1] - time[0]
    dict = {}
    for behavior in ['foraging_vs_exploration', 'other_patch_related']:
        dict[behavior] = {'both_wheel_activated': 0, 'correct_wheel_activated': 0, 'wrong_wheel_activated': 0,
                          'total_time': 0}
    dict['separate_foraging'] = {'both_wheel_activated': 0, 'one_wheel_activated': 0, 'no_wheel_activated': 0,
                                 'total_time': 0}
    dict['other_non_patch_related'] = {'both_wheel_activated': 0, 'one_wheel_activated': 0, 'no_wheel_activated': 0,
                                       'total_time': 0}
    assert timeline_bin > angular_speeds_bin, \
        'this has been implemented for behavioral data recorded at lower frequency than wheel data '
    for i, t in enumerate(angular_speeds['Time'][:-1]):
        times_select = (time >= t) * (time <= angular_speeds['Time'][i + 1])
        behaviors = timeline_v2[times_select]
        for behavior in behaviors:
            if behavior is not None:
                tracking_data_completeness['total'] += 1
                if behavior[0] in ['foraging_vs_exploration', 'direct_competition', 'close_by', 'travel_towards',
                                   'travel_away']:
                    if behavior[0] == 'foraging_vs_exploration':
                        bhv = behavior[0]
                    else:
                        bhv = 'other_patch_related'
                    try:
                        dict[bhv]['correct_wheel_activated'] += timeline_bin * act_dict[behavior[2]][i] \
                                                                * (1 - act_dict[3 - behavior[2]][i])
                        dict[bhv]['wrong_wheel_activated'] += timeline_bin * act_dict[3 - behavior[2]][i] \
                            * (1 - act_dict[behavior[2]][i])
                        dict[bhv]['both_wheel_activated'] += timeline_bin * act_dict[behavior[2]][i] \
                                                             * act_dict[3 - behavior[2]][i]
                        dict[bhv]['total_time'] += timeline_bin
                        tracking_data_completeness[bhv] += 1
                    except KeyError as err:
                        if VERBOSE > 1:
                            print('Key error: ', err)
                            print('no tracking data for behavior ', behavior, ' at time ', t)
                else:
                    if behavior[0] == 'separate_foraging':
                        bhv = 'separate_foraging'
                    else:
                        bhv = 'other_non_patch_related'
                    w_a = ['no_wheel_activated', 'one_wheel_activated', 'both_wheel_activated'][act_dict[1][i]
                                                                                                + act_dict[2][i]]
                    dict[bhv][w_a] += timeline_bin
                    dict[bhv]['total_time'] += timeline_bin
    if VERBOSE > 0:
        print('Behavior vs wheel activation data done. Completeness score for DLC-wheel-discrimination: ')
        for behavior in ['foraging_vs_exploration', 'other_patch_related',
                         'separate_foraging', 'other_non_patch_related']:
            print(behavior, 100 * tracking_data_completeness[behavior] / tracking_data_completeness['total'], ' % ')
    return dict


def plot_behavior_vs_wheel_data(videos='1'):
    bars = {behavior: np.zeros(4) for behavior in ['foraging_vs_exploration', 'other_patch_related']}

    bars['other_non_patch_related'] = np.zeros(4)
    bars['separate_foraging'] = np.zeros(4)
    labels = {behavior: ['total time', 'correct wheel', 'wrong wheel']
              for behavior in ['foraging vs exploration', 'other (patch related)']}
    labels['other (non patch related)'] = ['total time', 'no wheel', 'one wheel', 'both wheel']
    labels['separate foraging'] = labels['other (non patch related)']

    for n in videos:
        time, timeline = create_timeline_v2(n)
        angular_speeds = compute_angular_speeds(n)
        threshold = 12
        dict = behavior_vs_wheel_activation(time, timeline, angular_speeds, threshold=threshold)
        for bhv in bars.keys():
            bars[bhv][0] += dict[bhv]['total_time']
            if bhv in ['foraging_vs_exploration', 'other_patch_related']:
                bars[bhv][1] += dict[bhv]['correct_wheel_activated']
                bars[bhv][2] += dict[bhv]['wrong_wheel_activated']
                bars[bhv][3] += dict[bhv]['both_wheel_activated']
            else:
                bars[bhv][1] += dict[bhv]['no_wheel_activated']
                bars[bhv][2] += dict[bhv]['one_wheel_activated']
                bars[bhv][3] += dict[bhv]['both_wheel_activated']
    for bhv in bars.keys():
        for k in range(1, 4):
            bars[bhv][k] *= 100/bars[bhv][0]
    bars['separate_foraging'][3] = 4
    return bars, labels


def compare_with_dlc(videos = '1'):
    bars = {behavior: 0 for behavior in ['foraging_vs_exploration', 'other_patch_related',  'separate_foraging',  'other_non_patch_related']}
    for n in videos:
        time, timeline = create_timeline_v2(n)
        timeline_bin = time[1] - time[0]
        occ_data = np.load('data_video_'+n+'.npz')
        for bhv in bars.keys():
            if bhv in BEHAVIORS:
                pass

characterize_angular_speed()
