from __future__ import with_statement
import json
import os
import tensorflow as tf
import cv2
import numpy as np

from shutil import copyfile



path = "/volumes/Bella_li"

def subsample_video_fps_test():
    try:
        os.makedirs(path + "/bdd100k/video5fpsTEST/train/")  # video1 is the directory where to put videos not ok with (2)
    except FileExistsError:
        print("directory already exists")

        pass

    try:

        video = cv2.VideoCapture(path + "/bdd100k/videos/train/00a0f008-3c67908e.mov")
        counter = 0
        #print(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        #size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(path + "/bdd100k/video5fpsTEST/train/00a0f008-3c67908e" + '.mp4', fourcc, 5.0, (1280,720))  # save videos in mp4

        while (video.isOpened()):
            counter += 1
            # print(video.get(4))
            # print(counter)
            ret, frame = video.read()

            if np.shape(frame) == ():  # to prevent error while EOF is reached

                break

            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('frame', gray)


            if counter % 6 ==0:
                out.write(frame)

        out.release()
        video.release()

    except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
        print("error in opening video file")



def open_video():


    # Create a VideoCapture object and read from input file

    #video = cv2.VideoCapture(path + "/bdd100k/videos/train/00a0f008-3c67908e.mov")
    video = cv2.VideoCapture(path + "/bdd100k/video5fpsTEST/train/00a0f008-3c67908e.mp4")
    #video.set(cv2.CAP_PROP_FPS, int(5))
    print("Frame rate : " + str(video.get(cv2.CAP_PROP_FPS)))
    counter = 0

    while (video.isOpened()):
        counter += 1
        #print(video.get(4))
        print(counter)
        ret, frame = video.read()

        #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)


        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('frame', gray)

        if np.shape(frame) == ():  # to prevent error while EOF is reached

            break

        cv2.imshow('frame', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    # When everything done, release the video capture object
    video.release()

    # Closes all the frames
    cv2.destroyAllWindows()
