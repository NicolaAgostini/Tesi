from Preprocessing_bdd100k import *
from test import *
from TFLoad_bdd100k import *
from Glove import Glove
import numpy as np



alpha = 0.2



def main():
    a = Glove("/Users/nicolago/Desktop/Glove.6B/", alpha, "standard")

    #print(a.find_similar("move")[1:6])
    b = a.get_ysoft()
    print(b.shape)

    a.print_heatmap()

    # TEST #
    test_nearest(b)





if __name__ == '__main__':
    main()