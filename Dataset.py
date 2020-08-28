from torch.utils import data
import pandas
import lmdb
import tqdm
import numpy as np
import csv
import re
from Main import root_path


def txt_to_csv(path_oftxt, which_part):
    """
    from the txt files return the (training/val/test) file csv in which there are annotations well formed for action
    :param path_oftxt:
    :return:
    """
    count = 0
    with open(path_oftxt, 'r') as f:
        with open(root_path+"egtea/"+which_part+".csv", 'w+', newline='') as file:
            for line in f:

                values = re.split("-| ", line)
                v_name = values[0]+"-"+values[1]+"-"+values[2]
                # start_time = values[3]  # Not used
                # end_time = values[4]  # Not used
                start_sec = str(int(values[3]) / 1000)

                end_sec = str(int(values[4]) / 1000)

                action_id = int(values[7])-1

                writer = csv.writer(file)
                writer.writerow([count, v_name, start_sec, end_sec, action_id])

                count += 1
    return root_path+"egtea/"+which_part+".csv"


def read_representations(frames, env):
    """ Reads a set of representations, given their frame names and an LMDB environment.
    Applies a transformation to the features if provided
    :return features :  an array of images for a specific modality containing the desired images of the sequence
    """
    features = []
    # for each frame
    for f in frames:
        # read the current frame
        with env.begin() as e:
            dd = e.get(f.strip().encode('utf-8'))  # method to read .mdb file and save in dd the raw image
        if dd is None:
            print(f)  # frame not found
            print("NOT FOUND")
        # convert to numpy array
        data = np.frombuffer(dd, 'float32')
        # append to list
        features.append(data)
    # convert list to numpy array
    features = np.array(features)

    return features  # dim 14 X 1024 for rgb and flow and 352 for obj

def read_data(frames, env):
    """A wrapper form read_representations to handle loading from more environments.
    This is used for multimodal data loading (e.g., RGB + Flow)
    :return l : a list of 3 arrays: rgb, opt flow, obj containing the frames (img) for the considered sequence
    """

    # read the representations from all environments
    l = [read_representations(frames, e) for e in env]
    return l


class Dataset(data.Dataset):
    """
    this class will provide and object dataset which implement the methods __getitem()__ and __len()__
    and give an iterator on the dataset composed by features of rgb, optical flow and objects
    """
    def __init__(self, lmdb_path, groundthruth_path_csv):
        # read all videos annotations csv in pandas
        self.allvideos = pandas.read_csv(groundthruth_path_csv, header=None,
                                         names=["id", "video", "start", "end", "action"])
        self.lmdb_path = lmdb_path
        self.img_tmpl = "frame_{:010d}.jpg"  # the template used to save the frames in the features .mdb file

        self.alphaa = 0.25
        self.sequence_length = 14
        self.fps = 30

        self.ids = []  # action ids
        self.discarded_ids = []  # list of ids discarded (e.g., if there were no enough frames before the beginning of the action
        self.past_frames = []  # names of frames sampled before each action
        self.labels = []  # labels of each action

        self.env = [lmdb.open(l, readonly=True, lock=False) for l in self.lmdb_path]

        #print("How many environments? " + str(len(self.env)))

        self.handlevideos()

    def get_frames(self, frames, video):
        """ format file names using the image template """
        frames = np.array(list(map(lambda x: video + "_" + self.img_tmpl.format(x), frames)))
        return frames

    def handlevideos(self):
        """
        get the past_frames of the videos and assign them a label for the action id
        """
        for index, value in tqdm.tqdm(self.allvideos.iterrows(), total=len(self.allvideos)):
            frames = self.sample_frames_past(value.start)

            if frames.min() >= 1:  # if the smaller frame is at least 1, the sequence is valid

                # given the timestamp of each video snippets V_i save in past_frames the name  string corresponding to
                # the frames associated with my sequence
                self.past_frames.append(self.get_frames(frames, value.video))
                # now past frames are the list in correct format of the frames in lmdb

                self.ids.append(value.id)  # append the id of the video

                self.labels.append(value.action)

            else:  # if the sequence is invalid then insert in the list of invalid sequence

                self.discarded_ids.append(value.id)
        print("DISCARDED "+str(len(self.discarded_ids)))



    def sample_frames_past(self, point):
        """
        Samples frames before the beginning of the action "point", frames is an array of timestamps of the video where
        the video snippet V_i starts
        """
        # generate the relative timestamps, depending on the requested sequence_length
        # e.g., 2.  , 1.75, 1.5 , 1.25, 1.  , 0.75, 0.5 , 0.25
        # in this case "2" means, sample 2s before the beginning of the action
        time_stamps = np.arange(self.alphaa, self.alphaa * (self.sequence_length + 1), self.alphaa)[::-1]  # reverse order

        # compute the time stamp corresponding to the beginning of the action, it is in seconds
        end_time_stamp = point
        # subtract time stamps to the timestamp of the last frame
        time_stamps = end_time_stamp - time_stamps

        # convert timestamps to frames
        # use floor to be sure to consider the last frame before the timestamp (important for anticipation!)
        # and never sample any frame after that time stamp
        frames = np.floor(time_stamps * self.fps).astype(int)

        # sometimes there are not enough frames before the beginning of the action
        # in this case, we just pad the sequence with the first frame
        # this is done by replacing all frames smaller than 1
        # with the first frame of the sequence
        if frames.max() >= 1:
            frames[frames < 1] = frames[frames >= 1].min()

        return frames

    def __len__(self):
        """
        override of the len method of dataset pytorch
        :return: len of the dataset
        """
        return len(self.ids)

    def __getitem__(self, index):
        """
        override of get item method of dataset pytorch
        :param index: the index of the sequence to analyze
        :return out: a dict where id is the id of action_label.csv sequence, past_features are the ....
                     and label is the action to predict
        """
        # get past frames
        past_frames = self.past_frames[index]

        # return a dictionary containing the id of the current sequence
        out = {'id': self.ids[index]}

        # read representations for past frames
        out['past_features'] = read_data(past_frames, self.env)
        #print(out["id"])  # print the id of the sequence

        # get the label of the current sequence
        label = self.labels[index]
        out['label'] = label  # label is from 0 to 105

        return out
