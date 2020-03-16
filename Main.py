from Dataset import *
from test import *
from TFLoad import *




def main():
    #path = "/Users/nicolago/Desktop/test/*.mp4"

    #a = Dataset(5, "mph", path, 24, 16)

    # count_items(path + "/bdd100k/traingroundtruth.json", True)

    #a.preprocess_dataset()
    #open_video()
    #subsample_video_fps_test()




    #load_datas(path)

    #a.generate_img("/Users/nicolago/Desktop/test/train/", 16)

    #convert_toarray("/Users/nicolago/Desktop/test/imgs/")

    #load_datas()
    #a.generate_img("/Users/nicolago/Desktop/test/train/", "train")


    #load_dataset()
    read_tfrecord()
    #read_tfrecord("/Users/nicolago/Desktop/test/img.tfrecords")
    #image, classlabel = read_tfrecord("/Users/nicolago/Desktop/test/0.tfrec")

    #cose("/Users/nicolago/Desktop/test/0.tfrec")


if __name__ == '__main__':
    main()