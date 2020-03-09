import cv2
import os
from tqdm import tqdm
import numpy as np



def subsample_video_fps(root_path, path_from, path_to, which_part):  # works on video file
    """
    :param root_path: string where you have the folder "bdd100k"
    :param path_from: string where to find videos to process
    :param path_to: string where to put the processed videos
    :param which_part: depending if processing "train" or "val"
    :return: the videos processed in mp4 and at 5 fps, rotated each frame of 90 degrees counterclockwise
    """
    try:
        os.makedirs \
            (root_path + path_to + which_part + "/")  # video1 is the directory where to put videos not ok with (2)
    except FileExistsError:
        print("directory already exists")

        pass
    print("Subsampling the videos in 5 fps")
    for v_name in tqdm(os.listdir(root_path + path_from + which_part)):
        if not v_name.startswith('.'):
            # print(v_name)  # v_name is the name of the video and also the name of Json which has velocity info about that video
            # info_name = v_name
            try:

                video = cv2.VideoCapture(root_path + path_from + which_part + "/" + v_name)
                counter = 0

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # conversion in mp4
                if os.path.exists(root_path + path_to + which_part + "/" + v_name + '.mp4'):
                    print(" Video already exists!! CHECK ME! ")
                out = cv2.VideoWriter \
                    (root_path + path_to + which_part + "/" + os.path.splitext(v_name)[0] + '.mp4', fourcc, 5.0, (1280, 720))  # save videos in mp4

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



                    if counter % 6 == 0:
                        out.write(frame)


                out.release()
                video.release()

            except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
                print("error in opening video file")