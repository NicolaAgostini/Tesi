from Dataset import *
from test import *




def main():
    a = Dataset(5, "mph", path, 24)

    # count_items(path + "/bdd100k/traingroundtruth.json", True)

    a.preprocess_dataset()
    #open_video()
    #subsample_video_fps_test()


if __name__ == '__main__':
    main()