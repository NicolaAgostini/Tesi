import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal
from PIL import Image, ImageDraw
from collections import namedtuple
import cv2


def _str2frame(frame_str, fps=None):
    if fps==None:
        fps = 24

    splited_time = frame_str.split(':')
    assert len(splited_time) == 4

    time_sec = 3600 * int(splited_time[0]) \
               + 60 * int(splited_time[1]) +  int(splited_time[2])

    frame_num = time_sec * fps + int(splited_time[3])

    return frame_num

def parse_gtea_gaze(filename, gaze_resolution=None):
    '''
    Read gaze file in CSV format
    Input: 
        name of a gaze csv file
    return 
        an array where the each row follows: 
        (frame_num): px (0-1), py (0-1), gaze_type
    '''
    if gaze_resolution is None:
        # gaze resolution (default 1280*960)
        gaze_resolution = np.array([960, 1280], dtype=np.float32)

    # load all lines
    lines = [line.rstrip('\n') for line in open(filename)]
    # deal with different version of begaze
    ver = 1
    if '## Number of Samples:' in lines[9]:
        line = lines[9]
        ver = 1
    else:
        line = lines[10]
        ver = 2

    # get the number of samples
    values = line.split()
    num_samples = int(values[4])

    # skip the header
    lines = lines[34:]

    # pre-allocate the array 
    # (Note the number of samples in header is not always accurate)
    num_frames = 0
    gaze_data = np.zeros((num_samples*2, 4), dtype=np.float32)

    # parse each line
    for line in lines:
        values = line.split()
        # read gaze_x, gaze_y, gaze_type and frame_number from the file
        if len(values)==7 and ver==1:
            px, py = float(values[3]), float(values[4])
            frame = int(values[5])
            gaze_type = values[6]

        elif len(values)==26 and ver==2:
            px, py = float(values[5]), float(values[6])
            frame = _str2frame(values[-2])
            gaze_type = values[-1]

        else:
            raise ValueError('Format not supported')

        # avg the gaze points if needed
        if gaze_data[frame, 2] > 0:
            gaze_data[frame,0] = (gaze_data[frame,0] + px)/2.0
            gaze_data[frame,1] = (gaze_data[frame,1] + py)/2.0
        else:
            gaze_data[frame,0] = px
            gaze_data[frame,1] = py

        # gaze type
        # 0 untracked (no gaze point available); 
        # 1 fixation (pause of gaze); 
        # 2 saccade (jump of gaze); 
        # 3 unkown (unknown gaze type return by BeGaze); 
        # 4 truncated (gaze out of range of the video)
        if gaze_type == 'Fixation':
            gaze_data[frame, 2] = 1
        elif gaze_type == 'Saccade':
            gaze_data[frame, 2] = 2 
        else:
            gaze_data[frame, 2] = 3

        num_frames = max(num_frames, frame)

    gaze_data = gaze_data[:num_frames+1, :]

    # post processing:
    # (1) filter out out of bound gaze points
    # (2) normalize gaze into the range of 0-1
    for frame_idx in range(0, num_frames+1):

        px = gaze_data[frame_idx, 0] 
        py = gaze_data[frame_idx, 1]
        gaze_type = gaze_data[frame_idx, 2]

        # truncate the gaze points
        if (px < 0 or px > (gaze_resolution[1]-1)) \
           or (py < 0 or py > (gaze_resolution[0]-1)):
            gaze_data[frame_idx, 2] = 4

        px = min(max(0, px), gaze_resolution[1]-1)
        py = min(max(0, py), gaze_resolution[0]-1)

        # normalize the gaze
        gaze_data[frame_idx, 0] = px / gaze_resolution[1]
        gaze_data[frame_idx, 1] = py / gaze_resolution[0]
        gaze_data[frame_idx, 2] = gaze_type            

    return gaze_data

def plot_gaze():
    """Sample for gaze IO"""
    # gaze type
    gaze_type = ['untracked', 'fixation', 'saccade', 'unknown', 'truncated']

    # old version
    test_file_01 = '/Users/nicolago/Desktop/gaze_data/gaze_data/OP01-R01-PastaSalad.txt'
    test_data_01 = parse_gtea_gaze(test_file_01)



    start1 = int((754360/1000)-(3.5))
    end1 = int(754360 /1000)
    x = []
    y = []
    Frames = np.arange(start1*24, end1*24, 1)
    image_name = "/Volumes/Bella_li/frames/OP01-R01-PastaSalad/OP01-R01-PastaSalad_frame_{:010d}.jpg"

    for i in Frames:
        x.append(test_data_01[i,0])
        y.append(test_data_01[i,1])
        plt.scatter(x=[test_data_01[i,0]], y=[test_data_01[i,1]],  c='r', s=40)
    plt.show()

    """
    which_frame = 50
    print('Frame {:d}, Gaze Point ({:02f}, {:0.2f}), Gaze Type: {:s}'.format(
        Frames[which_frame],
        test_data_01[Frames[which_frame], 0],
        test_data_01[Frames[which_frame], 1],
        gaze_type[int(test_data_01[Frames[which_frame], 2])]
    ))
    im = Image.open(image_name.format(int((Frames[which_frame] / 24) * 30)))
    width, height = im.size
    im = plt.imread(image_name.format(int((Frames[which_frame] / 24) * 30)))
    implot = plt.imshow(im)
    plt.scatter(x=[test_data_01[Frames[which_frame], 0]*width], y=[test_data_01[Frames[which_frame], 1]*height], c='r', s=100)
    """
    #print(x)

    #plt.plot(np.asarray(x), np.asarray(y), color='black')
    #plt.savefig("Gaze.jpg")
    # print the loaded gaze
    """
    print('Loaded gaze data from {:s}'.format(test_file_01))
    print('Frame {:d}, Gaze Point ({:02f}, {:0.2f}), Gaze Type: {:s}'.format(
            1000, 
            test_data_01[1000, 0], 
            test_data_01[1000, 1], 
            gaze_type[int(test_data_01[1000, 2])]
        ))
    
    # new version
    test_file_02 = './gaze_data/P16-r03-BaconAndEggs.txt'
    test_data_02 = parse_gtea_gaze(test_file_02)
    # print the loaded gaze
    print('Loaded gaze data from {:s}'.format(test_file_02))
    print('Frame {:d}, Gaze Point ({:02f}, {:0.2f}), Gaze Type: {:s}'.format(
            1000, 
            test_data_02[1000, 0], 
            test_data_02[1000, 1], 
            gaze_type[int(test_data_02[1000, 2])]
        )) 
    """

def return_gaze_point(index_fr, file):
    """

    :param index_fr: the index of the frame to get gaze info (index is in 30 fps format)
    :param file: name of file gaze
    :return: two cordinates of gaze x,y normalized on the image size
    """
    """Sample for gaze IO"""
    # gaze type
    gaze_type = ['untracked', 'fixation', 'saccade', 'unknown', 'truncated']

    # old version
    test_file_01 = '/home/2/2014/nagostin/Desktop/gaze_data/gaze_data/'+file+'.txt'
    test_data_01 = parse_gtea_gaze(test_file_01)

    list = os.listdir("/home/2/2014/nagostin/Desktop/frames/" + file)  # dir is your directory path
    number_files = len(list)


    f = signal.resample(test_data_01, number_files)

    return (f[index_fr][0]),(f[index_fr][1])


def get_gaze_mask(gaze_point, image_size, mask_kind='soft', radius=60):
    '''
    Args:
        input gaze_point: named tuple point thas stores x and y
        input image_size: tuple that contains the image size (h, w)
        input mask_kind: string in ['soft', 'hard']
        input radius: radius of the circle in the mask

    '''

    # Retrieve image shape
    h, w = image_size

    # Soft mask
    if mask_kind == 'soft':
        def super_gaussian(x, center=None, radius=60):
            def super_gaussian_i(x_i, center, radius):
                if len(x_i.shape) == 1:
                    x_i = x_i.reshape(-1, 1)
                x_i = x_i - center
                x_i = np.exp(- (np.dot(x_i.T, x_i) / (70 * radius)) ** 8)
                return x_i

            # Retrieve input shape
            N, x_dim = x.shape

            # Mean and sigma
            if center is None:
                center = np.zeros([x_dim, 1])
            if isinstance(center, list) or isinstance(center, tuple):
                center = np.array(center).reshape(x_dim, 1)

            # Compute the bell function
            x = np.array([super_gaussian_i(x_i, center=center, radius=radius) for x_i in x]).reshape(N, 1)
            return x

        num_points = 50
        x_points = np.linspace(0, w, num_points)
        y_points = np.linspace(0, h, num_points)
        xv, yv = np.meshgrid(x_points, y_points)
        mask = np.stack([yv, xv], axis=-1).reshape(-1, 2)
        mask = super_gaussian(mask, center=[gaze_point.y, gaze_point.x], radius=radius).reshape(num_points, num_points)
        mask = mask / mask.max()
        mask = cv2.resize(mask, (w, h))
        mask = np.stack([mask, mask, mask], axis=-1)

    # Hard mask
    elif mask_kind == 'hard':
        mask = Image.new('RGB', (w, h))
        draw = ImageDraw.Draw(mask)
        draw.ellipse((gaze_point.x - radius, gaze_point.y - radius, gaze_point.x + radius, gaze_point.y + radius),
                     fill=(255, 255, 255))
        mask = np.array(mask)
        mask = mask / mask.max()

    else:
        raise Exception(f'Error. Mask kind {mask_kind} not supported.')

    return mask


##################################################
# Main
##################################################

def return_cropped_img(image, gaze_x, gaze_y, h, w, raggio, type = "hard"):
    """
    return the images cropped following the mask specification and the gaze point
    :param image: the image to be cropped as a numpy array
    :param gaze_x: the x point of the gaze
    :param gaze_y:
    :param h: the height of the image
    :param w:
    :param raggio: the radius of the gaze cropping
    :type type: type of label smoothing
    :return: the image cropped following the gaze
    """

    # Set the gaze point
    Point = namedtuple('Point', ['x', 'y'])
    gaze_point = Point(x=gaze_x, y=gaze_y)

    # Create mask
    if type == "hard":
        mask_hard = get_gaze_mask(gaze_point, image_size=(h, w), mask_kind='hard', radius=raggio)
        # Create image with gaze
        image_gaze = image * mask_hard
    elif type == "soft":
        mask_soft = get_gaze_mask(gaze_point, image_size=(h, w), mask_kind='soft', radius=raggio)
        image_gaze = image * mask_soft

    return image_gaze





def test_gaze(file = "OP01-R02-TurkeySandwich"):
    """
    A simple function to test the gaze data
    :param file:
    :return:
    """

    test_file_01 = '/Users/nicolago/Desktop/gaze_data/gaze_data/' + file + '.txt'
    test_data_01 = parse_gtea_gaze(test_file_01)
    test_data_01 = np.asarray(test_data_01)

    print(test_data_01.shape)

    list = os.listdir("/Volumes/Bella_li/frames/"+file)  # dir is your directory path
    number_files = len(list)
    print(number_files)

    f = signal.resample(test_data_01, number_files)

    print(f.shape)


    return 0

