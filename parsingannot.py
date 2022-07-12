from sys import argv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir


# file = open(argv[1], mode='r', encoding="utf-8-sig")
file = open('annot_files/mouse1_session1_001.annot' , mode='r', encoding="utf-8-sig")
lines = file.readlines()
file.close()


def get_movie_file(annot):
    movie_file_line = annot[1]
    char = 0
    while char < len(movie_file_line) and movie_file_line[char] != '/':
        char += 1
    return movie_file_line[char:]


def get_annot_framerate(annot):
    annot_framerate_line = annot[6]
    char = 0
    while char < len(annot_framerate_line) and annot_framerate_line[char] != ':':
        char += 1
    return float(annot_framerate_line[char+2:])


def get_annot_start_time(annot):
    annot_start_time_line = annot[4]
    char = 0
    while char < len(annot_start_time_line) and annot_start_time_line[char] != ':':
        char += 1
    return float(annot_start_time_line[char+2:])


def get_annot_end_time(annot):
    annot_end_time_line = annot[5]
    char = 0
    while char < len(annot_end_time_line) and annot_end_time_line[char] != ':':
        char += 1
    return float(annot_end_time_line[char + 2:])


def behavior_parser(annot):
    filepath = get_movie_file(annot)
    annot_framerate = get_annot_framerate(annot)
    l = 0  # line count
    behaviors = {}
    while l < len(annot):
        if annot[l][0] == '>':
            behavior = annot[l][1:-1]
            behavior_starts = []
            behavior_ends = []
            behavior_durations = []
            l += 2
            while len(annot[l]) > 1:
                char = 0
                start = ''
                end = ''
                duration = ''
                while annot[l][char] in '1234567890':
                    start += annot[l][char]
                    char += 1
                char += 1
                while annot[l][char] in '1234567890':
                    end += annot[l][char]
                    char += 1
                char += 1
                while annot[l][char] in '1234567890':
                    duration += annot[l][char]
                    char += 1
                    if char == len(annot[l]):
                        break
                behavior_starts.append(float(start))
                behavior_ends.append(float(end))
                behavior_durations.append(float(duration))
                l += 1
            l += 1
            behaviors[behavior] = {'start': behavior_starts, 'end': behavior_ends, 'duration': behavior_durations}
        else:
            l += 1
    return behaviors


def get_sessid(annot):
    file = get_movie_file(annot)
    char = 0
    while char < len(file) and file[char] != 'B':
        char += 1
    sessid = file[char:-6]
    return sessid


def get_threshold_change(sessid):
    patch1 = pd.read_csv('sessionsdata/state1_' + sessid + '.csv')
    patch2 = pd.read_csv('sessionsdata/state2_' + sessid + '.csv')
    if len(patch2) > len(patch1):
        patch = patch1
        patchid = 1
    else :
        patch = patch2
        patchid = 2
    row = 0
    initial_threshold = patch.iloc[0]['threshold']
    while row < len(patch) and patch.iloc[row]['threshold'] == initial_threshold :
        row += 1
    if row < len(patch) :
        return {'patchid': patchid,
                'changetime': patch.iloc[row]['time_in_session'],
                'init_threshold': initial_threshold,
                'new_threshold': patch.iloc[row]['threshold'] }
    else:
        return None


def get_positions(sessid):
    datetime = pd.to_datetime(sessid.split('_')[1], utc=True)
    pos_csv = []
    for file in listdir('sessionsdata/'):
        if file[:8] == 'FrameTop':
            datetimef = pd.to_datetime(file.split('_')[1], utc=True)
            if abs(datetimef - datetime) < pd.Timedelta(10, 'm'):
                pos_csv.append(file)
    if len(pos_csv) > 1:
        raise Exception("too many position csv files found for sessid :", sessid, "files found: ", files)
    elif len(pos_csv) == 0:
        raise Exception("no csv file found for sessid:", sessid)
    else :
        file = pos_csv[0]
        datetimef = pd.to_datetime(file.split('_')[1], utc=True)
        positions = pd.read_csv('sessionsdata/'+file)
        offset = (datetimef - datetime).total_secondes()
        positions.rename(columns = {'Unnamed: 0':'frame'}, inplace = True)
        positions = positions.assign(Time=positions.frame/50 + offset)   #timestamp
        return positions


def discriminate_behaviours(behaviors, sessid):
    individuals_behaviors = {'ind1': {behavior: {} for behavior in behaviors.keys()},
                             'ind2': {behavior: {} for behavior in behaviors.keys()}}
    name_mapping = {0 : 'ind1', 1: 'ind2'}
    positions = get_positions(sessid)
    behavior_fr = 30
    for behavior in behaviors:
        if (behavior == 'attack' or behavior == 'investigation' or behaviour == 'separate_exploration'
                or behaviour == 'separate_foraging'):
            individuals_behaviors['ind1'][behavior] = behaviors[behavior]
        elif (behavior == 'travel_towards' or behavior == 'travel_away' or behavior == 'direct_competition'
            or behavior == 'close_by' or behavior == 'foraging_vs_exploration'):
            for i, start in enumerate(behaviors[behavior]['start']):
                start = start/behavior_fr
                end = behaviors[behavior]['end'][i]/behavior_fr
                pos = positions.iloc[list(start <= positions.Time <= end)]
                ind1_dist_to_patches = [(pos.ind1_nose_x.mean() - pos.single_right_wheel_x.mean()) ** 2
                + (pos.ind1_nose_y.mean() - pos.single_right_wheel_y.mean()) ** 2 ,
                                         (pos.ind1_nose_x.mean() - pos.single_left_wheel_x.mean()) ** 2
                + (pos.ind1_nose_y.mean() - pos.single_left_wheel_y.mean()) ** 2]
                ind2_dist_to_patches = [(pos.ind2_nose_x.mean() - pos.single_right_wheel_x.mean()) ** 2
                + (pos.ind2_nose_y.mean() - pos.single_right_wheel_y.mean()) ** 2 ,
                                         (pos.ind2_nose_x.mean() - pos.single_left_wheel_x.mean()) ** 2
                + (pos.ind2_nose_y.mean() - pos.single_left_wheel_y.mean()) ** 2]
                ind, patch = np.where(
                    [ind1_dist_to_patches,
                     ind2_dist_to_patches] == np.min([ind1_dist_to_patches, ind2_dist_to_patches]))
                individuals_behaviors[name_mapping[ind]][behavior]





