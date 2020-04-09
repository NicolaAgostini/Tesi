from Preprocessing_bdd100k import *
from test import *
from TFLoad_bdd100k import *
from Glove import Glove
import numpy as np



alpha = 0.8



def main():
    a = Glove("/Users/nicolago/Desktop/Glove.6B/", alpha)
    a.load_glove()
    #print(a.find_similar("move")[1:6])
    phi = a.compute_phi()
    print(phi.shape)
    a.compute_Pi(phi)
    #a.print_heatmap()
    print(a.get_onehot())
    b = a.get_ysoft()
    test_nearest(b)



if __name__ == '__main__':
    main()