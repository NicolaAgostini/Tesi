import numpy as np
import lmdb
from tqdm import tqdm
import os


path_of_dir = "/home/2/2014/nagostin/Desktop/"

### CODE TO EXTRACT THE NEW FEATURE HAND OBJECT INTERACTION HOI ###

for file in os.listdir(path_of_dir+"newfeat"):
    print("Processing "+ file)
    env = lmdb.open(path_of_dir+'hand_obj_newfeat', map_size=1099511627776)
    video_name = file.split("_")[0] + '_frame_{:010d}.jpg'
    detections = np.load(path_of_dir+"newfeat/"+file, allow_pickle=True, encoding='bytes')
    n_feat = 0
    for i, dets in enumerate(tqdm(detections,'Extracting features')):
        feat = np.zeros(54, dtype='float32')
        for index,value in enumerate(dets):
            feat[index] = value
        #feat = dets
        #print(feat.shape)
        key = video_name.format(i+1)
        #print(key)
        n_feat = n_feat + np.count_nonzero(feat)



        with env.begin(write=True) as txn:
            txn.put(key.encode(),feat)

    #print("not zero feat = " + str(n_feat))
