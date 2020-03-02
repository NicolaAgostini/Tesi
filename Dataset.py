from __future__ import with_statement
import json
import os
import tensorflow as tf
import cv2
import numpy as np

from shutil import copyfile
from test import *
from tqdm import tqdm



#path = os.getcwd()
path = "/volumes/Bella_li"


class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

class Dataset():
    """A simple class for handling data sets."""

    def __init__(self, vel, format, path_data):

        self.vel = vel
        self.format = format
        self.path_data = path_data



    def data_files(self):

        tf_record_pattern = os.path.join(FLAGS.data_dir, '%s-*' % self.subset)
        data_files = tf.gfile.Glob(tf_record_pattern)
        if not data_files:
            print('No files found for dataset %s/%s at %s' % (self.name,
                                                            self.subset,
                                                            FLAGS.data_dir))

            self.download_message()
            exit(-1)
        return data_files

    def reader(self):

      return tf.TFRecordReader()

    def visualize(self, net_inputs, net_outputs):
    # input a batch of training examples of form [Tensor1, Tensor2, ... Tensor_n]
    # net_inputs: usually images; net_outputs: usually the labels
    # this function visualize the data that is read in, do not return anything but use tf.summary
      raise NotImplemented()

    def get_threshold(self):

        if self.format == "mph":
            threshold = 3.11
        else:
            if self.format == "kmh":
                threshold = 5
            else:
                raise InputError("You must declare the format properly : choose from mph and kmh")
        return threshold



    def subsample_video_fps(self, which_part):  # works on video file
        """
        :param which_part: string "train" or "val" which are the possible split of dataset
        :return: the videos subsampled at 5 fps in ( path + "/bdd100k/video5fps/" + which_part + "/" )
        """
        try:
            os.makedirs(path + "/bdd100k/video5fps/" + which_part + "/")  # video1 is the directory where to put videos not ok with (2)
        except FileExistsError:
            print("directory already exists")

            pass
        print("Subsampling the videos in 5 fps")
        for v_name in tqdm(os.listdir(path + "/bdd100k/video30fps/" + which_part)):
            if not v_name.startswith('.'):
                #print(v_name)  # v_name is the name of the video and also the name of Json which has velocity info about that video
                # info_name = v_name
                try:

                    video = cv2.VideoCapture(self.path_data + "/bdd100k/video30fps/" + which_part + "/" + v_name)
                    counter = 0

                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    if os.path.exists(path + "/bdd100k/video5fps/" + which_part + "/" + v_name + '.mp4'):
                        print(" Video already exists!! CHECK ME! ")
                    out = cv2.VideoWriter(path + "/bdd100k/video5fps/" + which_part + "/" + os.path.splitext(v_name)[0] + '.mp4', fourcc, 5.0, (1280, 720))  #save videos in mp4

                    while (video.isOpened()):
                        counter += 1
                        #print(video.get(4))
                        #print(counter)
                        ret, frame = video.read()

                        if np.shape(frame) == ():  # to prevent error while EOF is reached

                            break

                        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)


                        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        # cv2.imshow('frame', gray)



                        if counter % 6 == 0:
                            out.write(frame)


                    out.release()
                    video.release()

                except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
                    print("error in opening video file")






    def move_video(self, video_path, threshold):  # help function for clean_dataset()
        """
        :param video_path:
        :param threshold:
        :return:
        """

        print("Selecting videos and json file in which the occurence of stopping event is after 10 sec since the beggining of the video")

        for v_name in tqdm(os.listdir(path + video_path)):  # reading all the videos file
            if not v_name.startswith('.'):
                #print(v_name)  # v_name is the name of the video and also the name of Json which has velocity info about that video
                info_name = os.path.splitext(v_name)[0] + ".json"
                try:
                    if os.path.exists(path + '/bdd100k/info/100k/train/' + info_name):
                        with open(path + '/bdd100k/info/100k/train/' + info_name, 'r') as f:
                            info = json.load(f)

                            seq_ok = True
                            i = 0
                            for v in info["locations"]:  # not consider sequences where car vel < 5 in the first 10 sec

                                if v["speed"] < threshold and i <= 10:
                                    seq_ok = False
                                    break
                                i += 1

                            if seq_ok == True:
                                if "train" in video_path:
                                    if not os.path.exists(path + "/bdd100k/video30fps/train/" + v_name):
                                        copyfile(path + "/bdd100k/videos/train/" + v_name,
                                                path + "/bdd100k/video30fps/train/" + v_name)  # copy video in     path + "/bdd100k/video30fps/train/
                                else:
                                    if "val" in video_path:
                                        if not os.path.exists(path + "/bdd100k/video30fps/val/" + v_name):
                                            copyfile(path + "/bdd100k/videos/val/" + v_name,
                                                     path + "/bdd100k/video30fps/val/" + v_name)  # copy video in     path + "/bdd100k/video30fps/val/

                                if not os.path.exists(path + "/bdd100k/info1/100k/train/" + info_name):
                                    copyfile(path + "/bdd100k/info/100k/train/" + info_name,
                                             path + "/bdd100k/info1/100k/train/" + info_name)  # copy info json in    path + "/bdd100k/info1/100k/train/

                    else:
                        with open(path + '/bdd100k/info/100k/val/' + info_name, 'r') as f:
                            info = json.load(f)

                            seq_ok = True
                            i = 0
                            for v in info["locations"]:  # not consider sequences where car vel < 5 in the first 10 sec

                                if v["speed"] < threshold and i <= 10:
                                    seq_ok = False
                                    break
                                i += 1

                            if seq_ok == True:
                                if "train" in video_path:
                                    if not os.path.exists(path + "/bdd100k/video30fps/train/" + v_name):
                                        copyfile(path + "/bdd100k/videos/train/" + v_name,
                                                 path + "/bdd100k/video30fps/train/" + v_name)  # copy video in     path + "/bdd100k/video30fps/train/
                                else:
                                    if "val" in video_path:
                                        if not os.path.exists(path + "/bdd100k/video30fps/val/" + v_name):
                                            copyfile(path + "/bdd100k/videos/val/" + v_name,
                                                     path + "/bdd100k/video30fps/val/" + v_name)  # copy video in     path + "/bdd100k/video30fps/val/

                                if not os.path.exists(path + "/bdd100k/info1/100k/val/" + info_name):
                                    copyfile(path + "/bdd100k/info/100k/val/" + info_name,
                                             path + "/bdd100k/info1/100k/val/" + info_name)

                except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
                    print("error in opening Json Info file")

    def clean_dataset(self, which_part):  # works on video file
        """
        :param which_part:
        :return: the dataset (video and info files) without the videos where the car stopped before 10 secs
        """


        try:
            os.makedirs(
                path + "/bdd100k/video30fps/" + which_part + "/")  # video1 is the directory where to put videos not ok with (2)
        except FileExistsError:
            print("Warning " + "video30fps/" + which_part + "directory already exists")

            pass
        try:
            os.makedirs(path + "/bdd100k/info1/100k/train")
        except FileExistsError:
            print("Warning " + "/bdd100k/info1/100k/" + which_part + "directory already exists")

            pass

        try:
            os.makedirs(path + "/bdd100k/info1/100k/val")
        except FileExistsError:
            print("Warning " + "/bdd100k/info1/100k/" + "val" + "directory already exists")

            pass

        threshold = self.get_threshold()



        self.move_video("/bdd100k/videos/" + which_part , threshold)




    def generate_groundtruth(self, which_part):  # only works on Json in info1 must be executed after generated info1 folder and its files
        """
        :param which_part: values "train" or "val"
        :return: the ground truth file created ONLY for specific which_part, if which_part is "val" the the file regards the TEST subset
        """
        try:
            os.makedirs(path + "/bdd100k/groundtruth/train/")
        except FileExistsError:
            print("directory already exists")

            pass
        try:
            os.makedirs(path + "/bdd100k/groundtruth/test")
        except FileExistsError:
            print("directory already exists")

            pass

        threshold = self.get_threshold()

        data = {}
        data["videos"] = []
        print("generating groundtruth file for all the videos")
        for info_name in tqdm(os.listdir(path + "/bdd100k/info/100k/" + which_part + "/")):
            #print(info_name)
            if not info_name.startswith('.'):

                try:
                    with open(path + "/bdd100k/info/100k/" + which_part + "/"+info_name, 'r') as f:
                        try:
                            info = json.load(f)
                            i = 0
                            has_stopped = False
                            not_consider = False
                            for v in info["locations"]:  # not consider sequences where car vel < 5 in the first 10 sec

                                if v["speed"] < threshold:
                                    if i < 10:
                                        not_consider = True
                                        break  # i don't consider the file if stopped in less than 10 secs
                                        #raise InputError("ERROR: event of stopping in less than 10 secs")
                                    data["videos"].append({
                                        "filename": os.path.splitext(info_name)[0],
                                        "stop": i
                                    })
                                    has_stopped = True
                                    break

                                i += 1
                            if has_stopped == False and not_consider == False:
                                data["videos"].append({
                                    "filename": os.path.splitext(info_name)[0],
                                    "stop": "No"
                                })
                        except ValueError:  # includes simplejson.decoder.JSONDecodeError
                            print("Decoding JSON:  "+info_name+" has failed")

                        except InputError as error:
                            print('A New Exception occured: ', error.message)




                except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
                        print("error in opening Json Info file")
        if "val" in which_part:
            which_part = "test"
        with open(path + "/bdd100k/groundtruth/" + which_part + "/info.json", 'w+') as outfile:
            json.dump(data, outfile)


    def handle_positiveframes(self, which_part):  # prerequisites: the videos have to be subsampled at 5 fps and groundtruth.json file available
        for i in range(3):
            if "val" in which_part:
                try:
                    os.makedirs(path + "/bdd100k/" + "test" + "/positivi" + (i + 1))
                except FileExistsError:
                    print("directory already exists")
                    pass

                if i == 0:

                    try:
                        os.makedirs(path + "/bdd100k/" + "test" + "/negativi")
                    except FileExistsError:
                        print("directory already exists")
                        pass
            else:
                try:
                    os.makedirs(path + "/bdd100k/"+"train" + "/positivi"+(i+1))
                except FileExistsError:
                    print("directory already exists")
                    pass

                if i == 0:

                    try:
                        os.makedirs(path + "/bdd100k/"+"train" + "/negativi")
                    except FileExistsError:
                        print("directory already exists")
                        pass

        for i in range(3):
            try:
                with open(path + "/bdd100k/groundtruth/" + which_part + "/info.json", 'r') as f:
                    try:
                        info = json.load(f)


                        for v in info["videos"]:  # not consider sequences where car vel < 5 in the first 10 sec

                            if v["stop"] != "No":

                                if v["stop"] >=10*(i+1) and v["stop"] < 10*(i+2):

                                    name = v["filename"]
                                    if "train" in which_part:
                                        if not os.path.exists(path + "/bdd100k/train/positivi" + str(i+1) + "/" + name + ".mp4"):
                                            copyfile(path + "/bdd100k/video5fps/" + which_part+"/" + name + ".mp4",
                                                     path + "/bdd100k/train/positivi" + str(i+1) + "/" + name + ".mp4")
                                    else:
                                        if "val" in which_part:
                                            if not os.path.exists(path + "/bdd100k/test/positivi" + str(i+1) + "/" + name + ".mp4"):
                                                copyfile(path + "/bdd100k/video5fps/" + which_part + "/" + name + ".mp4",
                                                         path + "/bdd100k/test/positivi" + str(i+1) + "/" + name + ".mp4")

                            else:
                                if v["stop"] == "No":
                                    name = v["filename"]
                                    if "train" in which_part:
                                        if not os.path.exists(
                                                path + "/bdd100k/train/negativi" + "/" + name + ".mp4"):
                                            copyfile(path + "/bdd100k/video5fps/" + which_part + "/" + name + ".mp4",
                                                     path + "/bdd100k/train/negativi" + "/" + name + ".mp4")
                                    else:
                                        if "val" in which_part:
                                            if not os.path.exists(
                                                    path + "/bdd100k/test/negativi" + "/" + name + ".mp4"):
                                                copyfile(path + "/bdd100k/video5fps/" + which_part + "/" + name + ".mp4",
                                                         path + "/bdd100k/test/negativi" + "/" + name + ".mp4")







                    except ValueError:  # includes simplejson.decoder.JSONDecodeError
                        print("opening JSON:  info.json has failed")

            except InputError as error:
                print('A New Exception occured: ', error.message)






def main():

    a = Dataset(5, "kmh", path)
    #open_video()

    #a.clean_dataset("train")
    #a.clean_dataset("val")

    a.subsample_video_fps("train")
    a.subsample_video_fps("val")

    #a.generate_groundtruth("train")
    #a.generate_groundtruth("val")

    #a.handle_positiveframes("train")
    #a.handle_positiveframes("val")






if __name__ == '__main__':
    main()