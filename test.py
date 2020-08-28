from __future__ import with_statement
import cv2
from Model import *
from Dataset import *
from torch.utils.data import DataLoader



path = "/volumes/HD"



import subprocess
import shlex
import json


# SOME TESTS ON THE PROJECTS #

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


def test_nearest(y_soft):
    dict = {}
    for i in range(y_soft.shape[0]):
        dict[i] = y_soft[4][i]

    ordered = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    print(ordered)


def inspect_lmdb(path):
    env = lmdb.open(path, readonly=True, lock=False)
    with env.begin() as txn:
        with txn.cursor() as curs:
            # do stuff
            for key, value in curs:
                print((key, value))

"""
AN EXAMPLE OF WHAT THERE IS INSIDE LMDB FILES:

(b'OP01-R01-PastaSalad_frame_0000007548.jpg', b'\x00\x00\x00\x00\x01\x95\x05;9.\xd0>\x82\xfc!?n\xf3 ETC....
(b'OP01-R01-PastaSalad_frame_0000007549.jpg', b'\xc97\xc0=3k\xed>\x00\x00\x00\x00\xba\x9a\x82?\x99\ ETC...

ETC...



"""

def read_mock_representation(e):
    if "obj" in e:
        return torch.randn(14, 352)
    else:
        return torch.randn(14, 1024)


def read_mock_data(env):
    l = [read_mock_representation(e) for e in env]
    return l


class Mock_Dataset(data.Dataset):
    """
    this class will provide and object dataset which implement the methods __getitem()__ and __len()__
    and give an iterator on the dataset composed by features of rgb, optical flow and objects
    """

    def __init__(self, env):
        self.env = env  # list of enviroments es. rgb, flow, obj
        self.ids =[1, 2]
        self.label = [56, 34]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        return a item mock out with label and past features
        """


        # return a dictionary containing the id of the current sequence
        out = {'id': self.ids[index]}


        # read representations for past frames
        out['past_features'] = read_mock_data(self.env)

        # get the label of the current sequence

        out['label'] = self.label[index]

        return out

def get_mock_dataloader():
    a = Mock_Dataset(["rgb", "flow", "obj"])
    #print(a.__getitem__(1))
    return DataLoader(a, batch_size=2)  # change batch size


def test_model():
    """
    function to test the input and output size of the model
    """
    device = torch.device("cpu")

    batch_size = 1
    seq_len = 14
    input_dim = [1024, 1024, 352]

    model = BaselineModel(1, seq_len, input_dim)  # batch size, temporal sequence, input dimension
    model.to(device)

    loaders = get_mock_dataloader()

    for i, batch in enumerate(loaders):
        x = batch['past_features']
        #print(len(x))

        y = batch['label']

        output = model(x)
        #output = model(dict_in)
        print(output.size())



def print_data(path_1):

    env = lmdb.open(path_1, readonly=True, lock=False)
    with env.begin() as e:
        with e.cursor() as curs:
            print(curs.item())
            dd = e.get("P05-R01-PastaSalad_frame_0000020636.jpg".encode('utf-8'))
        data = np.frombuffer(dd, 'float32')
        print(data)
    #read_representations()


    detections = np.load("/aulahomes2/2/2014/nagostin/Desktop/video/" + "P05-R01-PastaSalad_detections.npy",
                         allow_pickle=True, encoding='bytes')
    print(detections[20635])










