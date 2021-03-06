import sys
from os.path import expanduser, exists
import deeplabcut
from os import listdir
from parsingannot import *
import shutil

'''
# As long as you have cloned the aeon_mecha_de folder into
# repos in your home filter
sys.path.append(expanduser('~/repos/aeon_mecha_de'))

# This path is only true if you are on pop.erlichlab.org
dataroot = '/home/gregoiredy/mnt/delab/data/arena0.1/socialexperiment0_raw/'
figpath = '/home/gregoiredy/mnt/delab/figures/'
exportpath = '/home/gregoiredy/repos/behavior_analysis/sessionsdata'


import numpy as np
import pandas as pd
import aeon.analyze.patches as patches
import os
import aeon.preprocess.api as api
import matplotlib.pyplot as plt

import aeon.util.helpers
import aeon.util.plotting
helpers = aeon.util.helpers
plotting = aeon.util.plotting

sessdf = api.sessiondata(dataroot)
sessdf = api.sessionduration(sessdf)                                     # compute session duration
sessdf = sessdf[~sessdf.id.str.contains('test')]
sessdf = sessdf[~sessdf.id.str.contains('jeff')]
sessdf = sessdf[~sessdf.id.str.contains('OAA')]
sessdf = sessdf[~sessdf.id.str.contains('rew')]
sessdf = sessdf[~sessdf.id.str.contains('Animal')]
sessdf = sessdf[sessdf.loc[:, 'start'] > np.datetime64("2022-01-01")]

sessdf.reset_index(inplace=True, drop=True)

df = sessdf.copy()

valid_id_file = expanduser('~/mnt/delab/conf/valid_ids.csv')
vdf = pd.read_csv(valid_id_file)
valid_ids = list(vdf.id.values[vdf.real.values == 1])


df.id = df.id.apply(helpers.fixID, valid_ids=valid_ids)
helpers.mergeSocial(df)
helpers.merge(df)
df = df[df.id.str.contains(';')]
df = df[df.id.str.contains('706')]  # we want 1 black and 1 bleached mouse


def exportDataToCSV(limit=1e6):
    done = 0
    for session in df.itertuples():  # This is easily parallelized :shrug:
        print(f'Exporting {helpers.getSessionID(session)}')
        helpers.exportWheelData(dataroot, session, datadir=exportpath, format='csv', force=False)
        if done >= limit:
            return
        else:
            done += 1
'''

dest = '/home/gregoiredy/repos/behavior_analysis/sessionsdata/'
config_path = '/home/gregoiredy/dlc_out_for_gregoire/dlc_project/config.yaml'


for n in '2':
    video_path = '/home/gregoiredy/to_annotate/video' + str(n) + '/'
    filename = np.load('data_video_' + str(n) + '.npz')['movie_file'].item()
    filename = filename.split('/')[-1]
    filename = filename.split(' ')[0]
    deeplabcut.analyze_videos(config_path, [video_path + filename], save_as_csv=True, destfolder=dest)




