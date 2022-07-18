from sys import argv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir


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
        raise Exception("too many position csv files found for sessid :", sessid, "files found: ", [f for f in pos_csv])
    elif len(pos_csv) == 0:
        raise Exception("no csv file found for sessid:", sessid)
    else :
        file = pos_csv[0]
        datetimef = pd.to_datetime(file.split('_')[1], utc=True)
        positions = pd.read_csv('sessionsdata/'+file)
        offset = (datetimef - datetime).total_seconds()
        positions.rename(columns = {'Unnamed: 0':'frame'}, inplace = True)
        positions = positions.assign(Time=positions.frame/50 + offset)   #timestamp
        return positions


def discriminate_behaviours(behaviors, positions):
    behavior_fr = 30
    for behavior in behaviors:
        if behavior in ['travel_towards', 'travel_away', 'direct_competition', 'close_by', 'foraging_vs_exploration']:

            behaviors[behavior]['individual_close_from_patch'] = []
            behaviors[behavior]['patch_of_interest'] = []

            for i, start in enumerate(behaviors[behavior]['start']):
                startt = start / behavior_fr
                if startt < positions.Time[0] or startt > positions.Time[len(positions.Time) - 1]:
                    ind, patch = None, None
                else:
                    end = behaviors[behavior]['end'][i]
                    endt = end/behavior_fr
                    pos = positions.iloc[list((startt <= positions.Time) & (positions.Time <= endt))]


                    ind1_dist_to_patches = [(pos.ind1_nose_x.mean() - pos.single_right_wheel_x.mean()) ** 2
                    + (pos.ind1_nose_y.mean() - pos.single_right_wheel_y.mean()) ** 2,
                                             (pos.ind1_nose_x.mean() - pos.single_left_wheel_x.mean()) ** 2
                    + (pos.ind1_nose_y.mean() - pos.single_left_wheel_y.mean()) ** 2]
                    ind2_dist_to_patches = [(pos.ind2_nose_x.mean()
                                             - pos.single_right_wheel_x.mean()) ** 2
                                            + (pos.ind2_nose_y.mean() - pos.single_right_wheel_y.mean()) ** 2,
                                            (pos.ind2_nose_x.mean() - pos.single_left_wheel_x.mean()) ** 2
                                            + (pos.ind2_nose_y.mean() - pos.single_left_wheel_y.mean()) ** 2]

                    ind, patch = np.where(
                        [ind1_dist_to_patches,
                         ind2_dist_to_patches] == np.min([ind1_dist_to_patches, ind2_dist_to_patches]))
                    if ind.size == 1:
                        ind = ind.item() + 1
                    else:
                        ind = None
                    if patch.size == 1:
                        patch = patch.item() + 1
                    else:
                        patch = None
                behaviors[behavior]['individual_close_from_patch'].append(ind)
                behaviors[behavior]['patch_of_interest'].append(patch)
    return behaviors


def make_videos_dicts():
    for n in '123456':
        f = open('annot_files/mouse1_session1_00' + n + '.annot', mode='r', encoding="utf-8-sig")
        l = f.readlines()
        f.close()
        bhv = behavior_parser(l)
        fr = get_annot_framerate(l)
        a_start = get_annot_start_time(l)
        a_end = get_annot_end_time(l)
        id = get_sessid(l)
        tc = get_threshold_change(id)
        try:
            pos = get_positions(id)
            bhv = discriminate_behaviours(bhv, pos)
        except FileNotFoundError:
            pass
        mv_file = get_movie_file(l)
        video_dict = {'id': id,
                      'movie_file': mv_file,
                      'annot_fr': fr,
                      'annot_start': a_start,
                      'annot_end': a_end,
                      'threshold_change': tc,
                      'behavior_data': bhv,
                      'pos_data': pos}
        np.savez('data_video_' + str(n), **video_dict)


make_videos_dicts()

