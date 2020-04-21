from Preprocessing_bdd100k import *
from test import *
from TFLoad_bdd100k import *
from Glove import Glove
import numpy as np
from Dataset import *
from torch.utils.data import DataLoader



alpha = 0.2

path_to_lmdb = ["/Volumes/Bella_li/egtea/TSN-C_3_egtea_action_CE_s1_flow_model_best_fcfull_hd/",
                    "/Volumes/Bella_li/egtea/TSN-C_3_egtea_action_CE_s1_rgb_model_best_fcfull_hd/"]

groundtruth_path = "/volumes/Bella_li/egtea/action_annotation/train_split1.txt"

groundtruth_path_csv = "/volumes/Bella_li/egtea/training.csv"


def get_dataset(batch_size, num_workers):
    a = Dataset(path_to_lmdb, groundtruth_path_csv)
    return DataLoader(a, batch_size=batch_size, num_workers=num_workers,
                      pin_memory=True, shuffle= True) # suffle true for training


def main():


    #inspect_lmdb(path_to_lmdb[1])
    #txt_to_csv(groundtruth_path)






def label_smmothing(set_modality="standard", alpha=0.1, temperature = 0):
    """
    :param set_modality: standard or softmax or prior
    :param alpha:
    :param temperature: provided for softmax
    :return: smoothed labels depending on the specific modality
    """
    a = Glove("/Users/nicolago/Desktop/Glove.6B/", alpha, set_modality, temperature)

    # print(a.find_similar("move")[1:6])
    b = a.get_ysoft()
    return b


if __name__ == '__main__':
    main()
