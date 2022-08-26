from typing import Any

import numpy as np
import pandas as pd
from os import listdir
from os.path import exists
from math import sqrt
pos_data_dir = '/home/gregoiredy/mnt/delab/data/arena0.1/socialexperiment0_preprocessed/'


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
    return float(annot_framerate_line[char + 2:])


def get_annot_start_time(annot):
    annot_start_time_line = annot[4]
    char = 0
    while char < len(annot_start_time_line) and annot_start_time_line[char] != ':':
        char += 1
    return float(annot_start_time_line[char + 2:])


def get_annot_end_time(annot):
    annot_end_time_line = annot[5]
    char = 0
    while char < len(annot_end_time_line) and annot_end_time_line[char] != ':':
        char += 1
    return float(annot_end_time_line[char + 2:])


def behavior_parser(annot):
    lin: int = 0  # line count
    behaviors = {}
    while lin < len(annot):
        if annot[lin][0] == '>':
            behavior = annot[lin][1:-1]
            behavior_starts = []
            behavior_ends = []
            behavior_durations = []
            lin += 2
            while len(annot[lin]) > 1:
                char = 0
                start = ''
                end = ''
                duration = ''
                while annot[lin][char] in '1234567890':
                    start += annot[lin][char]
                    char += 1
                char += 1
                while annot[lin][char] in '1234567890':
                    end += annot[lin][char]
                    char += 1
                char += 1
                while annot[lin][char] in '1234567890':
                    duration += annot[lin][char]
                    char += 1
                    if char == len(annot[lin]):
                        break
                behavior_starts.append(float(start))
                behavior_ends.append(float(end))
                behavior_durations.append(float(duration))
                lin += 1
            lin += 1
            behaviors[behavior] = {'start': behavior_starts, 'end': behavior_ends, 'duration': behavior_durations}
        else:
            lin += 1
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
    else:
        patch = patch2
        patchid = 2
    row = 0
    initial_threshold = patch.iloc[0]['threshold']
    while row < len(patch) and patch.iloc[row]['threshold'] == initial_threshold:
        row += 1
    if row < len(patch):
        return {'patchid': patchid,
                'changetime': patch.iloc[row]['time_in_session'],
                'init_threshold': initial_threshold,
                'new_threshold': patch.iloc[row]['threshold']}
    else:
        return None


def get_positions(sessid):  # TODO : fix this
    if sessid == 'BAA-1100704:BAA-1100706_2022-02-09T08:59:17':
        positions = pd.read_csv('sessionsdata/' + 'FrameTop_2022-02-09T09-00-00_position.csv')
        offset = 43
        positions.rename(columns={'Unnamed: 0': 'frame'}, inplace=True)
        positions = positions.assign(Time=positions.frame / 50 + offset)
        return positions

    datetime = pd.to_datetime(sessid.split('_')[1], utc=True)
    pos_csv = []
    for directory in listdir('sessionsdata/'):
        if directory[:8] == 'FrameTop':
            datetimef = pd.to_datetime(directory.split('_')[1], utc=True)
            if abs(datetimef - datetime) < pd.Timedelta(10, 'm'):
                pos_csv.append(listdir(directory)[0])
    if len(pos_csv) > 1:
        raise FileNotFoundError("too many position csv files found for sessid :", sessid,
                                "files found: ", [f for f in pos_csv])
    elif len(pos_csv) == 0:
        raise FileNotFoundError("no csv file found for sessid:", sessid)
    else:
        file = pos_csv[0]
        datetimef = pd.to_datetime(file.split('_')[1], utc=True)
        positions = pd.read_csv('sessionsdata/' + file)
        offset = (datetimef - datetime).total_seconds()
        positions.rename(columns={'Unnamed: 0': 'frame'}, inplace=True)
        positions = positions.assign(Time=positions.frame / 50 + offset)  # timestamp
        return positions


def get_wheel_data(sessid):
    if exists('sessionsdata/wheel1_' + sessid + '.csv'):
        right_wheel = pd.read_csv('sessionsdata/wheel1_' + sessid + '.csv')
    else:
        raise FileNotFoundError('wheel 1 data not found for session ', sessid)
    if exists('sessionsdata/wheel2_' + sessid + '.csv'):
        left_wheel = pd.read_csv('sessionsdata/wheel2_' + sessid + '.csv')
    else:
        raise FileNotFoundError('wheel 2 data not found for session ', sessid)

    wheel_dict = {}
    for side, w_data in zip(['right', 'left'], [right_wheel, left_wheel]):
        w_data.rename(columns={'time_in_session': 'Time'}, inplace=True)
        wheel_dict[side] = w_data
    return wheel_dict


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
                    endt = end / behavior_fr
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


def get_occupancy(pos):

    ROI_distance_quotient = 0.17

    def belong_to_circle(ptx, pty, cx, cy, radius):
        """ Returns if a point belongs to a circle. """
        return (ptx - cx) ** 2 + (pty - cy) ** 2 < radius ** 2

    def is_on_patch(patch1_x, patch1_y, patch2_x, patch2_y, ind_x, ind_y, radius):
        """ Returns i if the individual is on patch i """
        for i, p in enumerate([(patch1_x, patch1_y), (patch2_x, patch2_y)]):
            px, py = p
            if belong_to_circle(ind_x, ind_y, px, py, radius):
                return i + 1
        return 0

    df = pd.DataFrame(columns=['frame', 'patch_occupied_1', 'patch_occupied_2'])
    try:
        rwx, rwy, lwx, lwy = None, None, None, None
        frame = 0
        while sum([type(w) == np.float64 for w in (rwx, rwy, lwx, lwy)]) != 4:
            rwx = pos.single_right_wheel_x[frame]
            rwy = pos.single_right_wheel_y[frame]
            lwx = pos.single_left_wheel_x[frame]
            lwy = pos.single_left_wheel_y[frame]
            frame += 1
        distance_between_patches = sqrt((rwx - lwx)**2 + (rwy - lwy)**2)
        patch_radius = ROI_distance_quotient * distance_between_patches
        for i, frame in enumerate(pos.itertuples()):
            patch_occupied = [None, None]
            for k, ind in enumerate(['1', '2']):
                x = frame.ind1_nose_x
                y = frame.ind1_nose_y
                if x and y:
                    patch_occupied[k] = is_on_patch(rwx, rwy, lwx, lwy, x, y, patch_radius)
            df = pd.concat([df, pd.DataFrame({'frame': i, 'patch_occupied_1': patch_occupied[0],
                                              'patch_occupied_2': patch_occupied[1]}, index=[0])], ignore_index=True)
    except Exception as err:
        print(err)
    finally:
        return df


def make_videos_dicts():
    for n in '123456':
        f = open('annot_files/mouse1_session1_00' + n + '.annot', mode='r', encoding="utf-8-sig")
        lines = f.readlines()
        f.close()
        bhv = behavior_parser(lines)
        fr = get_annot_framerate(lines)
        a_start = get_annot_start_time(lines)
        a_end = get_annot_end_time(lines)
        id = get_sessid(lines)
        tc = get_threshold_change(id)
        try:
            pos = get_positions(id)
            bhv = discriminate_behaviours(bhv, pos)
        except FileNotFoundError:
            print('pos data not found for video ', n)
            pos = None
        if pos is not None:
            occupancy = get_occupancy(pos)
        wheel_data = get_wheel_data(id)
        mv_file = get_movie_file(lines)
        video_dict = {'id': id,
                      'movie_file': mv_file,
                      'annot_fr': fr,
                      'annot_start': a_start,
                      'annot_end': a_end,
                      'threshold_change': tc,
                      'behavior_data': bhv,
                      'pos_data': pos,
                      'wheel_data': wheel_data,
                      'occupancy': occupancy}
        np.savez('data_video_' + str(n), **video_dict)


make_videos_dicts()
