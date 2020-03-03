from __future__ import with_statement
import json
import os
import tensorflow as tf
import cv2
import numpy as np

from shutil import copyfile
from test import *
from tqdm import tqdm
import math
import random



#path = os.getcwd()
path = "/volumes/HD"


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

def count_items(file_path, is_json):
    if is_json == True:
        with open(file_path, 'r') as f:
            info = json.load(f)
            v = info["videos"]
            print("Total number of valid videos: "+ str(len(v)))
    else:
        print("Total number of files: " + str(sum([len(files) for r, d, files in os.walk(file_path)])))

class Dataset():
    """A simple class for handling data sets."""

    def __init__(self, vel, format, path_data, observed_frames):

        self.vel = vel
        self.format = format
        self.path_data = path_data
        self.observed_frames = observed_frames



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




    def generate_groundtruth(self, which_part):  # only works on Json in info
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
            os.makedirs(path + "/bdd100k/groundtruth/val")
        except FileExistsError:
            print("directory already exists")

            pass




        threshold = self.get_threshold()

        data = {}
        data["videos"] = []

        not_validdata = {}
        not_validdata["videos"] = []

        print("generating groundtruth file for all the videos")
        nnot_considered = 0
        nconsidered = 0
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
                                    if i <= 10:
                                        nnot_considered += 1
                                        not_consider = True
                                        not_validdata["videos"].append({
                                            "filename": os.path.splitext(info_name)[0],
                                            "stop": i
                                        })
                                        break  # i don't consider the file if stopped in less than 10 secs but i save to make the valid
                                        #raise InputError("ERROR: event of stopping in less than 10 secs")
                                    data["videos"].append({
                                        "filename": os.path.splitext(info_name)[0],
                                        "stop": i
                                    })
                                    nconsidered += 1
                                    has_stopped = True
                                    break

                                i += 1
                            if has_stopped == False and not_consider == False:
                                data["videos"].append({
                                    "filename": os.path.splitext(info_name)[0],
                                    "stop": "No"
                                })
                                nconsidered += 1
                        except ValueError:  # includes simplejson.decoder.JSONDecodeError
                            print("Decoding JSON:  "+info_name+" has failed")

                        except InputError as error:
                            print('A New Exception occured: ', error.message)




                except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
                        print("error in opening Json Info file")
        print("Number of invalid video is: " + str(nnot_considered))
        print("Number of valid video is: " + str(nconsidered))

        with open(path + "/bdd100k/groundtruth/" + which_part + "/info.json", 'w+') as outfile:
            json.dump(data, outfile)
        with open(path + "/bdd100k/groundtruth/" + which_part + "/infoNotValid.json", 'w+') as outfile:
            json.dump(not_validdata, outfile)


    def handle_positiveframes(self, which_part, validity):  # prerequisites:  groundtruth.json and rain_how_tocut.json files available
        """
        :param which_part:
        :param validity: value "valid" or "not" (for not valid) depending the videos is valid or not
        :return: this function uses groundtruth json and splits the videos in 3 positive parts and 1 negative
        """
        for i in range(3):
            if "val" in which_part:
                try:
                    os.makedirs(path + "/bdd100k/" + "val" + validity + "/positivi" + str(i + 1))
                except FileExistsError:
                    print("directory already exists")
                    pass

                if i == 0:

                    try:
                        os.makedirs(path + "/bdd100k/" + "val" + validity + "/negativi")
                    except FileExistsError:
                        print("directory already exists")
                        pass
            else:
                try:
                    os.makedirs(path + "/bdd100k/"+"train" + validity + "/positivi"+str(i+1))
                except FileExistsError:
                    print("directory already exists")
                    pass

                if i == 0:

                    try:
                        os.makedirs(path + "/bdd100k/"+"train" + validity + "/negativi")
                    except FileExistsError:
                        print("directory already exists")
                        pass

        for i in range(3):
            try:
                if "valid" in validity:
                    path_ofjson = path + "/bdd100k/groundtruth/" + which_part + "/info.json"
                else:
                    if "not" in validity:
                        path_ofjson = path + "/bdd100k/" +which_part+ "_how_tocut.json"

                with open(path_ofjson, 'r') as f:
                    try:
                        info = json.load(f)


                        for v in tqdm(info["videos"]):
                            name = v["filename"]

                            if os.path.exists(path + "/bdd100k/videos/" + which_part + "/" + name + ".mov"):
                                if v["stop"] != "No":

                                    if v["stop"] >= 10*(i+1) and v["stop"] < 10*(i+2):


                                        if "train" in which_part:
                                            if not os.path.exists(path + "/bdd100k/train"+ validity +"/positivi" + str(i+1) + "/" + name + ".mov"):
                                                copyfile(path + "/bdd100k/videos/" + which_part+"/" + name + ".mov",
                                                         path + "/bdd100k/train"+ validity +"/positivi" + str(i+1) + "/" + name + ".mov")
                                        else:
                                            if "val" in which_part:
                                                if not os.path.exists(path + "/bdd100k/val"+ validity +"/positivi" + str(i+1) + "/" + name + ".mov"):
                                                    copyfile(path + "/bdd100k/videos/" + which_part + "/" + name + ".mov",
                                                             path + "/bdd100k/val"+ validity +"/positivi" + str(i+1) + "/" + name + ".mov")

                                else:
                                    if v["stop"] == "No":

                                        if "train" in which_part:
                                            if not os.path.exists(
                                                    path + "/bdd100k/train"+ validity +"/negativi" + "/" + name + ".mov"):
                                                copyfile(path + "/bdd100k/videos/" + which_part + "/" + name + ".mov",
                                                         path + "/bdd100k/train"+ validity +"/negativi" + "/" + name + ".mov")
                                        else:
                                            if "val" in which_part:
                                                if not os.path.exists(
                                                        path + "/bdd100k/val"+ validity +"/negativi" + "/" + name + ".mov"):
                                                    copyfile(path + "/bdd100k/videos/" + which_part + "/" + name + ".mov",
                                                             path + "/bdd100k/val"+ validity +"/negativi" + "/" + name + ".mov")







                    except ValueError:  # includes simplejson.decoder.JSONDecodeError
                        print("opening JSON:  info.json has failed")

            except InputError as error:
                print('A New Exception occured: ', error.message)



    def invalid_tovalidJSON(self, which_part):
        """
        :prerequisites: works with groundtruth infoNotValid.json
        :return: this function will output a json, for invalid videos consisting of a json pointing where each filename
        (considered invalid) will be cut
        """
        threshold = self.get_threshold()
        how_tocut = {}
        how_tocut["videos"] = []
        try:
            with open(path + "/bdd100k/groundtruth/" + which_part + "/infoNotValid.json", 'r') as f:
                try:
                    info = json.load(f)  # has the FIRST time to stop
                    for v in tqdm(info["videos"]):
                        T0 = v["stop"]
                        #T1 = 0
                        with open(path + "/bdd100k/info/100k/" + which_part + "/" + v["filename"] + ".json", 'r') as infoFILE:
                            infoJsonvalues = json.load(infoFILE)  # has all the velocity within the 40 secs of the specific video file
                            i = 0
                            found = False
                            start = -1
                            end = -1
                            new_stop = -1
                            for q in infoJsonvalues["locations"]:
                                if q["speed"] < threshold:
                                    T0 = i


                                if (i - T0 - math.ceil(self.observed_frames / 5)) > 10 and found == False:  # if true it means that there are 10 sec without stopping event
                                    #interval = 10 + math.ceil(self.observed_frames / 5)
                                    found = True
                                    start = random.randint(T0, i)
                                    end = start + math.ceil(self.observed_frames/5) + 10 + 1
                                    if end > 40:
                                        start = T0
                                        end = start + math.ceil(self.observed_frames/5) + 10 + 1
                                        if end > 40:
                                            found = False
                                            break





                                if found==True and q["speed"] < threshold and i<= end:
                                    new_stop = i
                                    break


                                i += 1
                            if found == True:
                                if new_stop == -1:
                                    how_tocut["videos"].append({
                                        "filename": v["filename"],
                                        "start": start,
                                        "end": end,
                                        "stop": "No"
                                    })

                                else:
                                    how_tocut["videos"].append({
                                    "filename": v["filename"],
                                    "start": start,
                                    "end": end,
                                    "stop": new_stop
                                    })
                except ValueError:  # includes simplejson.decoder.JSONDecodeError
                    print("opening JSON:  info.json has failed")


        except InputError as error:
         print('A New Exception occured: ', error.message)


        with open(path + "/bdd100k/" + which_part + "_how_tocut.json", 'w+') as outfile:
            json.dump(how_tocut, outfile)













    def invalid_tovalid(self, which_part):
        """
        :return: this function will handle the invalid videos with infonotValid.json and create valid videos always in
        5 fps
        """
        self.invalid_tovalidJSON(which_part)





def main():

    a = Dataset(5, "mph", path, 16)
    #open_video()

    #a.clean_dataset("train")
    #a.clean_dataset("val")

    #a.subsample_video_fps("train")
    #a.subsample_video_fps("val")

    #a.generate_groundtruth("train")
    #a.generate_groundtruth("val")

    #a.handle_positiveframes("train")
    #a.handle_positiveframes("val")

    #count_items(path + "/bdd100k/groundtruth/test/info.json", True)

    #a.invalid_tovalidJSON("train")

    #a.invalid_tovalidJSON("val")


    a.handle_positiveframes("train", "valid")
    a.handle_positiveframes("val", "valid")

    a.handle_positiveframes("train", "not")
    a.handle_positiveframes("val", "not")






if __name__ == '__main__':
    main()