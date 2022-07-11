from sys import argv
import numpy as np
import matplotlib.pyplot as plt


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







