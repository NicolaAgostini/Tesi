import numpy as np
import lmdb
from tqdm import tqdm
import os


path_of_dir = "/home/2/2014/nagostin/Desktop/"

### OBJECT FEATURE EXTRACTION ###

"""
dict = {}
count=0

for file in os.listdir(path_of_dir+"video"):
    print("Processing "+ file)
    env = lmdb.open(path_of_dir+'obj_min', map_size=1099511627776)
    video_name = file.split("_")[0] + '_frame_{:010d}.jpg'
    detections = np.load(path_of_dir+"video/"+file, allow_pickle=True, encoding='bytes')
    
    for i, dets in enumerate(tqdm(detections,'Extracting features')):
        feat = np.zeros(56, dtype='float32')
        for d in dets:
            if(d[5]>0.5):
                if not int(d[0]) in dict:
                    dict[int(d[0])] = count
                    count += 1

                feat[dict[int(d[0])]]+=d[5]
        key = video_name.format(i+1)
        with env.begin(write=True) as txn:
            txn.put(key.encode(),feat)
    """
#standard feature creation of array of 352 for each frame
for file in os.listdir(path_of_dir+"video"):
    print("Processing "+ file)
    env = lmdb.open(path_of_dir+'obj_54_FT', map_size=1099511627776)
    video_name = file.split("_")[0] + '_frame_{:010d}.jpg'
    detections = np.load(path_of_dir+"video/"+file, allow_pickle=True, encoding='bytes')

    for i, dets in enumerate(tqdm(detections,'Extracting features')):
        feat = np.zeros(54, dtype='float32')
        #print(dets)
        for d in dets:
            print(d)
            feat[int(d[0])]+=d[5]
        key = video_name.format(i+1)
        with env.begin(write=True) as txn:
            txn.put(key.encode(),feat)



