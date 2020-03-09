from __future__ import with_statement
import json
import os
import tensorflow as tf
import cv2
import numpy as np
import time

from shutil import copyfile



path = "/volumes/HD"



import subprocess
import shlex
import json

def get_rotation(file_path_with_file_name):
    """
    Function to get the rotation of the input video file.
    Adapted from gist.github.com/oldo/dc7ee7f28851922cca09/revisions using the ffprobe comamand by Lord Neckbeard from
    stackoverflow.com/questions/5287603/how-to-extract-orientation-information-from-videos?noredirect=1&lq=1

    Returns a rotation None, 90, 180 or 270
    """
    cmd = "ffprobe -loglevel error -select_streams v:0 -show_entries stream_tags=rotate -of default=nw=1:nk=1"
    args = shlex.split(cmd)
    args.append(file_path_with_file_name)
    # run the ffprobe process, decode stdout into utf-8 & convert to JSON
    ffprobe_output = subprocess.check_output(args).decode('utf-8')
    if len(ffprobe_output) > 0:  # Output of cmdis None if it should be 0
        ffprobe_output = json.loads(ffprobe_output)
        rotation = ffprobe_output

    else:
        rotation = 0

    return rotation



def subsample_video_fps_test():



    video = cv2.VideoCapture(path + "/bdd100k/videos/train/000d4f89-3bcbe37a.mov")
    rot = get_rotation(path + "/bdd100k/videos/train/000d4f89-3bcbe37a.mov")

    print("ROTAZIONE: " + str(rot))
    counter = 0
    #print(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    #size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path + "/bdd100k/prove/ciao" + '.mp4', fourcc, 5.0, (1280,720))  # save videos in mp4

    print("width : " + str(video.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print("height : " + str(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

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



        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('frame', gray)


        if counter % 6 ==0:
            out.write(frame)

        counter += 1

    out.release()
    video.release()





def open_video():


    # Create a VideoCapture object and read from input file

    #video = cv2.VideoCapture(path + "/bdd100k/videos/train/00a0f008-3c67908e.mov")
    video = cv2.VideoCapture(path + "/bdd100k/videos/train/00ac3256-0f8e2cda.mov")
    #video.set(cv2.CAP_PROP_FPS, int(5))
    print("Frame rate : " + str(video.get(cv2.CAP_PROP_FPS)))

    print("width : " + str(video.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print("height : " + str(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    counter = 0

    while (video.isOpened()):
        counter += 1
        #print(video.get(4))
        #print(counter)
        ret, frame = video.read()

        if np.shape(frame) == ():  # to prevent error while EOF is reached

            break


        height, width = frame.shape[:2]

        if height > width:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)


        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('frame', gray)


        cv2.imshow('frame', frame)
        #time.sleep(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    # When everything done, release the video capture object
    video.release()

    # Closes all the frames
    cv2.destroyAllWindows()
