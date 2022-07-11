from parsingannot import *
import numpy as np
import sys
from os.path import expanduser


def get_behavior_pie_chart(annot):
    behaviors = behavior_parser(annot)
    annotated_behaviors = ['attack', 'close_by', 'direct_competition', 'foraging_vs_exploration',
                           'investigation', 'separate_exploration', 'separate_foraging', 'travel_away', 'travel_towards']

    behavior_total_durations = []
    for behavior in annotated_behaviors:
        behavior_total_durations.append(sum(behaviors[behavior]['duration']))
    return annotated_behaviors, behavior_total_durations


def label_values(pct):
    if pct < 2.5:
        return None
    else:
        return str(int(pct)) + '%'


def label_time(behavior_total_durations):
    time_labels = []
    for time in behavior_total_durations:
        if time//3600 > 0:
            time_labels.append('> ' + str(int(time // 3600)) + ' h')
        elif time//60 > 0:
            time_labels.append('> ' + str(int(time // 60)) + ' mn')
        else:
            time_labels.append('> ' + str(int(time)) + ' s')
    return np.array([time_labels])


def plot_pie_chart_time_repartition(ax, behaviors, behavior_total_durations, filename):
    behaviors_labels_dict = {'attack': 'attack', 'close_by': 'close by', 'direct_competition': 'direct competition',
                             'foraging_vs_exploration': 'foraging vs exploration', 'investigation': 'investigation',
                             'separate_exploration': 'separate exploration', 'separate_foraging': 'separate foraging',
                             'travel_away': 'travel away', 'travel_towards': 'travel towards'}
    behaviors_labels = [behaviors_labels_dict[behavior] for behavior in behaviors]
    behaviors_labels = np.array(behaviors_labels)
    labels = np.char.add(behaviors_labels, np.array([' ' for k in range(len(behaviors))]))
    labels = np.char.add(labels, label_time(behavior_total_durations))
    ax.pie(behavior_total_durations, labels=labels[0],
           autopct=lambda pct: label_values(pct),
           shadow=True, startangle=0, counterclock=False, radius=1.8)
    ax.set_title(label='total duration per behaviour, '+filename, y=-0.5)
    return ax


def total_time_variability_hist():
    total_time = np.zeros([6, 9])
    behaviors = ['attack', 'close\nby', 'direct\ncompetition', 'foraging vs\nexploration',
                 'investigation', 'separate\nexploration', 'separate\nforaging', 'travel\naway', 'travel\ntowards']
    for i, n in enumerate('123456'):
        file = open('annot_files/mouse1_session1_00' + n + '.annot', mode='r', encoding="utf-8-sig")
        lines = file.readlines()
        file.close()
        total_time[i] = np.array(get_behavior_pie_chart(lines)[1])
    std = np.std(total_time, axis=0)/np.mean(total_time, axis=0)
    return behaviors, std


def plot_time_variability_hist(ax, behaviors, std):
    ax.bar([1.5*k for k in range(9)], std, tick_label=behaviors, width=0.4)
    return ax


def create_timeline(annot):
    behaviors = behavior_parser(annot)
    annot_framerate = get_annot_framerate(annot)
    annot_start = get_annot_start_time(annot)
    annot_end = get_annot_end_time(annot)
    time = np.arange(start=annot_start, stop=annot_end, step=1/annot_framerate)
    timeline = np.empty(np.shape(time), dtype='<U30')
    for behavior in behaviors.keys():
        for k in range(len(behaviors[behavior]['start'])):
            behavior_start_frame = int(behaviors[behavior]['start'][k])
            behavior_end_frame = int(behaviors[behavior]['end'][k])
            timeline[behavior_start_frame: behavior_end_frame] = behavior
    return time, timeline


def get_mean_duration(behaviour, annot):
    behaviours = behavior_parser(annot)
    return np.mean(behaviours[behaviour]['duration']) / get_annot_framerate(annot)


def get_total_occurences(behaviour, annot):
    behaviours = behavior_parser(annot)
    return len(behaviours[behaviour]['start'])


def get_distribution_over_time(behaviour, annot, timebin=300, binstart=0, binstop=7200):
    behaviours = behavior_parser(annot)
    fr = get_annot_framerate(annot)
    bins = np.arange(start=binstart, stop=binstop, step=timebin)
    start_times = np.array(behaviours[behaviour]['start'])/fr
    return np.histogram(start_times, bins=bins)


def get_threshold_change_triggered_distribution(behaviour, annot, timebin=300, binstart=0, binstop=7200):
    first_time_stamp = 3727241957.4
    threshold_change_timestamp = 3727242919.38198    # for video 1 TODO: export from aeon
    threshold_change_time = threshold_change_timestamp - first_time_stamp
    behaviours = behavior_parser(annot)
    fr = get_annot_framerate()
    bins = np.arange(start=binstart-threshold_change_time, stop=binstop+threshold_change_time, step=timebin)
    start_times = np.array(behaviours[behaviour]['start']) / fr
    return np.histogram(start_times-threshold_change_time, bins=bins)


def get_next_behaviors_prob(behaviour, timeline):
    other_behaviors = ['attack', 'close_by', 'direct_competition', 'foraging_vs_exploration',
                       'investigation', 'separate_exploration', 'separate_foraging', 'travel_away', 'travel_towards']
    other_behaviors.remove(behaviour)
    next_behaviors_count = {b: 0 for b in other_behaviors}
    i = 0
    while i < np.shape(timeline)[0]:
        if timeline[i] == behaviour:
            while i + 1 < np.shape(timeline)[0] and (timeline[i] == behaviour or timeline[i] == ''):
                i += 1
            if timeline[i] != behaviour and timeline[i] != '':
                next_behaviors_count[timeline[i]] += 1
        else:
            i += 1
    for b in other_behaviors:
        next_behaviors_count[b] /= np.sum([next_behaviors_count[beh] for beh in other_behaviors])
    return next_behaviors_count


def get_previous_behaviour_prob(behaviour, timeline):
    return get_next_behaviors_prob(behaviour, np.flip(timeline))


def characterize_behavior(behavior):
    behaviors = ['attack', 'close_by', 'direct_competition', 'foraging_vs_exploration',
                 'investigation', 'separate_exploration', 'separate_foraging', 'travel_away', 'travel_towards']

    other_behaviors = behaviors.copy()
    other_behaviors.remove(behavior)
    mean_duration = []
    frequency = []
    distribution_over_time = []
    next_behaviors = {b: 0 for b in other_behaviors}
    previous_behaviors = {b: 0 for b in other_behaviors}

    tmax = []
    for n in '12345':
        file = open('annot_files/mouse1_session1_00' + n + '.annot', mode='r', encoding="utf-8-sig")
        lines = file.readlines()
        file.close()
        tmax.append(get_annot_end_time(lines))
    max_time = min(tmax)
    for n in '12345':

        file = open('annot_files/mouse1_session1_00' + n + '.annot', mode='r', encoding="utf-8-sig")
        lines = file.readlines()
        file.close()

        time, timeline = create_timeline(lines)

        mean_duration.append(get_mean_duration(behavior, lines))

        frequency.append(get_total_occurences(behavior, lines)/time[-1])

        distrib_over_time, bins = get_distribution_over_time(behavior, lines, binstop=5400)
        distribution_over_time.append(distrib_over_time)

        next_behaviors_prob = get_next_behaviors_prob(behavior, timeline)
        previous_behaviors_prob = get_previous_behaviour_prob(behavior, timeline)
        for b in other_behaviors:
            next_behaviors[b] += next_behaviors_prob[b]/5
            previous_behaviors[b] += previous_behaviors_prob[b]/5
    distribution_over_time = np.sum(distribution_over_time, axis=0)
    return {'mean_duration': np.mean(mean_duration),
            'frequency': np.mean(frequency),
            't_distrib': (distribution_over_time, bins),
            'next_behaviour_prob': next_behaviors,
            'prev_behaviour_prob': previous_behaviors}


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






