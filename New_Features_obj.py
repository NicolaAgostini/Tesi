import numpy as np
import lmdb
from tqdm import tqdm
import os


path_of_dir = "/home/2/2014/nagostin/Desktop/"

for file in os.listdir(path_of_dir+"newfeat"):
    print("Processing "+ file)
    env = lmdb.open(path_of_dir+'hand_obj_newfeat', map_size=1099511627776*2)
    video_name = file.split("_")[0] + '_frame_{:010d}.jpg'
    detections = np.load(path_of_dir+"newfeat/"+file, allow_pickle=True, encoding='bytes')

    for i, dets in enumerate(tqdm(detections,'Extracting features')):
        feat = dets
        #print(dets.shape)
        key = video_name.format(i+1)
        with env.begin(write=True) as txn:
            txn.put(key.encode(),feat)
