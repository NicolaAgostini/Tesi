import numpy as np
import lmdb
from tqdm import tqdm
import os


path_of_dir = "/home/2/2014/nagostin/Desktop/"

for file in os.listdir(path_of_dir+"featureobj"):
    print("Processing "+ file)
    env = lmdb.open(path_of_dir+'obj', map_size=1099511627776)
    video_name = file.split(".")[0] + '_frame_{:010d}.jpg'
    detections = np.load(path_of_dir+"featureobj/"+file, allow_pickle=True, encoding='bytes')

    for i, dets in enumerate(tqdm(detections,'Extracting features')):
        feat = np.zeros(352, dtype='float32')
        for d in dets:
            feat[int(d[0])]+=d[5]
        key = video_name.format(i+1)
        with env.begin(write=True) as txn:
            txn.put(key.encode(),feat)