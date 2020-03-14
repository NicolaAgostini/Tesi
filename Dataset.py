from __future__ import with_statement
import json
import os
import tensorflow as tf
import cv2
import numpy as np
import traceback

from shutil import copyfile
from test import *
from Video_ops import *
from tqdm import tqdm
import math
import random
import time



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
    """
    :param file_path: path of the file to count
    :param is_json: if True then the function will count the number of objects in a json
    :return: number of files recursively in a folder and subfolder else # of object in a json
    """
    if is_json == True:
        with open(file_path, 'r') as f:
            info = json.load(f)
            #v = info["videos"]
            print("Total number of valid videos: " + str(len(info)))
    else:
        print("Total number of files: " + str(sum([len(files) for r, d, files in os.walk(file_path)])))

class Dataset():
    """A simple class for handling data sets."""

    def __init__(self, vel, format, path_data, sample_size, frames_obs):

        self.vel = vel
        self.format = format
        self.path_data = path_data
        self.sample_size = sample_size
        self.frames_obs = frames_obs



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
        """
        :return: depending on the format, will return the threshold considered as Stopping Event
        """

        if self.format == "mph":
            threshold = 3.11
        else:
            if self.format == "kmh":
                threshold = 5
            else:
                raise InputError("You must declare the format properly : choose from mph and kmh")
        return threshold







    def change_videofps(self, which_part):  # works on video file
        """
        :param which_part: string "train" or "val" which are the possible split of dataset
        :return: the videos subsampled at 5 fps in ( path + "/bdd100k/video5fps/" + which_part + "/" )
        """
        subsample_video_fps("/volumes/HD", "/bdd100k/video30fps/", "/bdd100k/video5fps/", which_part)






    def move_video(self, video_path, threshold):  # help function for clean_dataset()
        """
        :param video_path: path
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
        :param which_part: depending if want to process "train" or "val" videos
        :return: the dataset (video and info files Json) without the videos where the car stopped before 10 secs
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

        self.move_video("/bdd100k/videos/" + which_part, threshold)




    def generate_groundtruth(self, which_part, path_to):  # only works on Json in info
        """
        :param which_part: depending is "train" or "val" data are considered
        :param path_to: where to punt the output json
        :return: info.json => for each video valid (DEF. Valid Video: velocity before the 10th sec >= threshold) the
        name of the video ("filename") and the time where the event of stop occurs ("stop")
          and infoNotValid.json => for each video not valid its name and where the event of stop occurs
        """
        try:
            os.makedirs(path + path_to + which_part + "/")
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

        with open(path + path_to + which_part + "/info.json", 'w+') as outfile:
            json.dump(data, outfile)
        with open(path + path_to + which_part + "/infoNotValid.json", 'w+') as outfile:
            json.dump(not_validdata, outfile)


    def order_videos(self, which_part, validity, path_to, path_groundtruth):  # prerequisites:  groundtruth.json and rain_how_tocut.json files available
        """
        :param which_part: "train" or "val"
        :param validity: "valid" or "not"
        :param path_to: where to store the folder containing the split of the videos
        :param path_groundtruth: path where is the groundtruth of each valid video
        :return: splits every videos, valid or not depending on validity variable, in one folder for negative examples
        (videos in which the car won't stop) and tree folders, depending on the interval in which the car will stop
        ("positivi1" if the car will stop between the 10th, included, and the 20th sec, excluded)
        """
        for i in range(3):

            try:
                os.makedirs(path + path_to + validity + "/positivi" + str(i + 1))
            except FileExistsError:
                print("directory already exists")
                pass

            if i == 0:

                try:
                    os.makedirs(path + path_to + validity + "/negativi")
                except FileExistsError:
                    print("directory already exists")
                    pass

        for i in range(3):
            try:
                if "valid" in validity:
                    path_ofjson = path + path_groundtruth + which_part + "/info.json"
                else:
                    if "not" in validity:
                        path_ofjson = path + "/bdd100k/temporal/" +which_part+ "_how_tocut.json"

                with open(path_ofjson, 'r') as f:
                    try:
                        info = json.load(f)


                        for v in tqdm(info["videos"]):
                            name = v["filename"]

                            if os.path.exists(path + "/bdd100k/videos/" + which_part + "/" + name + ".mov"):
                                if v["stop"] != "No":

                                    if v["stop"] >= 10*(i+1) and v["stop"] < 10*(i+2):

                                        if not os.path.exists(path + path_to + validity +"/positivi" + str(i+1) + "/" + name + ".mov"):
                                            copyfile(path + "/bdd100k/videos/" + which_part+"/" + name + ".mov",
                                                     path + path_to + validity +"/positivi" + str(i+1) + "/" + name + ".mov")

                                else:
                                    if v["stop"] == "No":

                                        if not os.path.exists(
                                                path + path_to + validity +"/negativi" + "/" + name + ".mov"):
                                            copyfile(path + "/bdd100k/videos/" + which_part + "/" + name + ".mov",
                                                     path + path_to + validity +"/negativi" + "/" + name + ".mov")

                    except ValueError:  # includes simplejson.decoder.JSONDecodeError
                        print("opening JSON:  info.json has failed")

            except InputError as error:
                print('A New Exception occured: ', error.message)



    def invalid_tovalidJSON(self, which_part):
        """
        :prerequisites: works with groundtruth infoNotValid.json
        :return: this function will output a json, for invalid videos only, consisting of a json pointing where each
        filename (considered invalid) will be cut. This json ($which_part + _how_tocut.json) in output has the following
        fields: "filename": the video's name
                "start": the second where to start cutting,
                "end": the second where to end cutting,
                "stop": "No" or the second in range ]start, end] where the car stops
        """
        threshold = self.get_threshold()
        how_tocut = {}
        how_tocut["videos"] = []
        try:
            with open(path + "/bdd100k/temporal/tgroundtruth/" + which_part + "/infoNotValid.json", 'r') as f:
                try:
                    info = json.load(f)  # has the FIRST time to stop
                    for v in tqdm(info["videos"]):
                        T0 = v["stop"]

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


                                if (i - T0 - math.ceil(self.sample_size / 5)) >= 10 and found == False:  # if true it means that there are 10 sec without stopping event
                                    #interval = 10 + math.ceil(self.observed_frames / 5)
                                    found = True
                                    start = random.randint(T0, i)  # start is a rondom number between the last stop and the second now considered
                                    end = start + math.ceil(self.sample_size/5) + 10
                                    if end > 40:
                                        start = T0
                                        end = start + math.ceil(self.sample_size/5) + 10
                                        if end > 40:
                                            found = False
                                            break





                                if found==True and q["speed"] < threshold and i <= end and i >= start + math.ceil(self.sample_size/5):
                                    new_stop = i  # update the stop time with a new stop time
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

        with open(path + "/bdd100k/temporal/" + which_part + "_how_tocut.json", 'w+') as outfile:
            json.dump(how_tocut, outfile)



    def handle_vid(self, which_part, validity, gt):  # prerequisites train/val + valid folders (3+1) and groundtruth/train/info.json available
        """
        :param which_part: "train" or "val"
        :return: for the VALID videos it returns the videos sampled at 5 fps, resized at 112 X 112 pixels and cut in
        observed frame prediction + 10 sec of prediction. It handles videos in train/val + valid folders
        Furthermore it cut the videos considered NOT-VALID following the specific of train/val + _how_tocut.json always
        maintaining the same specifics about height and width and fps
        """

        try:
            os.makedirs(path + "/bdd100k/" + which_part)
        except FileExistsError:
            print("directory already exists")
            pass


        try:
            if "valid" in validity:
                path_ofjson = path + "/bdd100k/temporal/tgroundtruth/" + which_part + "/info.json"
            else:
                if "not" in validity:
                    path_ofjson = path + "/bdd100k/temporal/" +which_part+ "_how_tocut.json"
            with open(path_ofjson, 'r') as f:
                try:
                    info = json.load(f)  # contains the stop events for each videos
                    number = -1
                    for v in tqdm(info["videos"]):
                        filename = v["filename"]

                        if v["stop"] == "No":
                            number = ""
                        else:
                            if int(v["stop"]) <= 20:
                                number = str(1)
                            else:
                                if int(v["stop"]) <= 30:
                                    number = str(2)
                                else:
                                    if int(v["stop"]) <= 40:
                                        number = str(3)

                        rot = -1
                        if number == "":
                            video = cv2.VideoCapture(path + "/bdd100k/temporal/train"+validity + "/negativi" + "/" + filename + ".mov")
                            if os.path.exists(path + "/bdd100k/temporal/train"+validity + "/negativi" + "/" + filename + ".mov"):
                                rot = get_rotation(path + "/bdd100k/temporal/train"+validity + "/negativi" + "/" + filename + ".mov")
                        else:
                            video = cv2.VideoCapture(path + "/bdd100k/temporal/train" + validity + "/positivi" + str(number) + "/" + filename + ".mov")
                            if os.path.exists(path + "/bdd100k/temporal/train" + validity + "/positivi" + str(number) + "/" + filename + ".mov"):
                                rot = get_rotation(path + "/bdd100k/temporal/train" + validity + "/positivi" + str(number) + "/" + filename + ".mov")

                        if video.isOpened() == False:
                            #print("Video NOT found")
                            continue
                        counter = 0

                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # conversion in mp4
                        if os.path.exists(path + "/bdd100k/" + which_part + "/" + filename + '.mp4'):
                            print(" Video already exists!! CHECK ME! ")
                            continue
                        out = cv2.VideoWriter(path + "/bdd100k/" + which_part + "/" + filename + '.mp4', fourcc, 5.0, (112, 112))  # save videos in mp4

                        while (video.isOpened()):

                            # print(video.get(4))
                            # print(counter)
                            ret, frame = video.read()

                            if np.shape(frame) == ():  # to prevent error while EOF is reached

                                break

                            if rot == 270:
                                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                            if rot == 90:
                                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                            if rot == -1:
                                print("mmmh")




                            if "valid" in validity:
                                if v["stop"] != "No":
                                    if counter % 6 == 0 and counter<10* (int(number)+1) *30 and counter>= 30*10*int(number) - ((self.sample_size/5)*30):  # every 6 frame is 1 sec (30fps original video)
                                        frame = cv2.resize(frame, (112, 112))
                                        out.write(frame)
                                else:
                                    if v["stop"] == "No":
                                        if counter % 6 == 0 and counter<20*30 and counter>= 30*10 - ((self.sample_size/5)*30):
                                            frame = cv2.resize(frame, (112, 112))
                                            out.write(frame)
                            else:
                                if "not" in validity:
                                    if v["stop"] != "No":
                                        if counter % 6 == 0 and counter < v["end"] * 30 and counter >= v["end"]*30 - 10*30 - (self.sample_size/5)*30:  # every 6 frame is 1 sec (30fps original video)
                                            frame = cv2.resize(frame, (112, 112))
                                            out.write(frame)
                                    else:
                                        if v["stop"] == "No":
                                            if counter % 6 == 0 and counter < v["end"] * 30 and counter >= v["end"]*30 - 10*30 - (self.sample_size/5)*30:
                                                frame = cv2.resize(frame, (112, 112))
                                                out.write(frame)

                            counter += 1  # HA SENSO  PARTIRE DALL' 1???

                        if v["stop"] == "No":
                            gt[filename] = "No"
                        else:
                            if v["stop"] != "No" and "valid" in validity:
                                gt[filename] = int(v["stop"]) - (10*int(number))  # the new stop is between 0 and 10 sec of the predicting time (not considering observed frames)
                            else:
                                if "not" in validity:
                                    gt[filename] = int(v["stop"]) - (int(v["end"]-10))

                        out.release()
                        video.release()



                except ValueError:  # includes simplejson.decoder.JSONDecodeError
                    print("Error:")
                    print(traceback.print_exc())

        except InputError as error:
            print('A New Exception occured: ', error.message)

        print(len(gt))
        time.sleep(3)
        return gt







    def preprocess_dataset(self):
        """
        :return: will output in the folder "/bdd100k" the videos cut and resized with groundtruth.json
        """

        gt = {}  # groundtruth


        self.generate_groundtruth("train", "/bdd100k/temporal/tgroundtruth/")
        self.generate_groundtruth("val", "/bdd100k/temporal/tgroundtruth/")

        self.invalid_tovalidJSON("train")
        self.invalid_tovalidJSON("val")

        self.order_videos("train", "valid", "/bdd100k/temporal/train", "/bdd100k/temporal/tgroundtruth/")
        self.order_videos("val", "valid", "/bdd100k/temporal/val", "/bdd100k/temporal/tgroundtruth/")
        
        self.order_videos("train", "not", "/bdd100k/temporal/train", "/bdd100k/temporal")
        self.order_videos("val", "not", "/bdd100k/temporal/val", "/bdd100k/temporal")



        gt = self.handle_vid("train", "valid", gt)

        gt = self.handle_vid("train", "not", gt)


        with open(path + "/bdd100k/" + "train" + "groundtruth.json", 'w+') as outfile:
            json.dump(gt, outfile)

        gt = {}

        gt = self.handle_vid("val", "valid", gt)
        gt = self.handle_vid("val", "not", gt)

        with open(path + "/bdd100k/" + "val" + "groundtruth.json", 'w+') as outfile:
            json.dump(gt, outfile)

        self.generate_img("volumes/HD/bdd100k/train/", "train")
        self.generate_img("volumes/HD/bdd100k/val/", "test")


    def generate_img(self, path, which_part):  # FIXME: ora lavoro nella cartella test, rendimi globale poi!!!!
        """
        :param path: path of the videos
        :param n_obs: number of observed frames
        :param which_part: if "train" or "test"
        :return: the images for every observed frame, in a specific folder
        """

        try:
            os.makedirs("/Users/nicolago/Desktop/test/"+which_part+"img/")
        except FileExistsError:
            print("directory already exists")
            pass

        not_considered = self.sample_size - self.frames_obs
        for i in tqdm(os.listdir(path)):
            with open("/Users/nicolago/Desktop/test/traingroundtruth.json", 'r') as f:
                info = json.load(f)
                if not i.startswith('.'):
                    try:
                        os.makedirs("/Users/nicolago/Desktop/test/"+which_part+"img/"+os.path.splitext(i)[0])
                    except FileExistsError:
                        print("directory already exists")
                        pass
                    vc = cv2.VideoCapture(path+i)
                    counter = 0
                    n_frame = 0
                    while (vc.isOpened()):

                        counter += 1

                        # print(video.get(4))
                        # print(counter)
                        ret, frame = vc.read()

                        if np.shape(frame) == ():  # to prevent error while EOF is reached

                            break



                        if counter > not_considered and counter<=self.sample_size:
                            n_frame += 1
                            cv2.imwrite("/Users/nicolago/Desktop/test/"+which_part+"img/"+os.path.splitext(i)[0]+"/"+os.path.splitext(i)[0]+"-"+str(n_frame)+".jpg", frame)
                    vc.release()










