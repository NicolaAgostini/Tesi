import torch
import os
from torch import nn
from pretrainedmodels import bninception
from torchvision import transforms
from glob import glob
from PIL import Image
import lmdb
from tqdm import tqdm
from os.path import basename
from argparse import ArgumentParser
from gaze_io_sample import *


### TO EXTRACT THE GAZED RGB FEATURE ###
#print(torch.__version__)

env = lmdb.open("/aulahomes2/2/2014/nagostin/Desktop/center_gaze", map_size=1099511627776)

for file in os.listdir("/aulahomes2/2/2014/nagostin/Desktop/frames/"):
    print(file)
    video_name = file.split("_")[0] + '_frame_{:010d}.jpg'
    f = gaze_arrays(file)
    for i,im in enumerate(tqdm(sorted(os.listdir("/aulahomes2/2/2014/nagostin/Desktop/frames/"+ file+"/")))):
        key = video_name.format(i+1)
        #gaze_center_x, gaze_center_y = return_gaze_point(i,file)  # sono normalizzati sulla grandezza dell'immagine
        gaze_center_x, gaze_center_y = f[i][0], f[i][1]
        center = [gaze_center_x,gaze_center_y]
        feat = np.asarray(center)
        with env.begin(write=True) as txn:
            txn.put(key.encode(),feat.tobytes())