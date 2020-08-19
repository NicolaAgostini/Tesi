import numpy as np
import os
import cv2
import pandas
import shutil
import random
import tqdm
import json
import glob
import sys
from scipy.spatial import distance
import matplotlib.pyplot as plt
import csv
import re


class ValueMeter(object):
    def __init__(self):
        self.sum = 0
        self.total = 0

    def add(self, value, n):
        self.sum += value*n
        self.total += n

    def value(self):
        return self.sum/self.total


class ArrayValueMeter(object):
    def __init__(self, dim=1):
        self.sum = np.zeros(dim)
        self.total = 0

    def add(self, arr, n):
        self.sum += arr*n
        self.total += n

    def value(self):
        val = self.sum/self.total
        if len(val) == 1:
            return val[0]
        else:
            return val


def topk_accuracy(scores, labels, ks, selected_class=None):
    """Computes TOP-K accuracies for different values of k
    Args:
        rankings: numpy ndarray, shape = (instance_count, label_count)
        labels: numpy ndarray, shape = (instance_count,)
        ks: tuple of integers
    Returns:
        list of float: TOP-K accuracy for each k in ks
    """


    rankings = scores.argsort()[:, ::-1]  # sort in ascending order where ranking[-1] is the scores index of max value,
                                          # then take everything but backwards so now rankings[0] is the score index of max value
    #print(rankings.shape)
    # trim to max k to avoid extra computation
    maxk = np.max(ks)
    #print(maxk)

    # compute true positives in the top-maxk predictions
    tp = rankings[:, :maxk] == labels.reshape(-1, 1)  # check where label from ranking (top 5 highest) is equal to groundtruth
    #print(tp)

    # trim to selected ks and compute accuracies
    return [tp[:, :k].max(1).mean() for k in ks]


def topk_accuracy_multiple_timesteps(preds, labels, ks=(1, 5)):
    accs = np.array(list(
        zip(*[topk_accuracy(preds[:, t, :], labels, ks) for t in range(preds.shape[1])])))
    return accs


def get_marginal_indexes(actions, mode):
    """For each verb/noun retrieve the list of actions containing that verb/name
        Input:
            mode: "verb" or "noun"
        Output:
            a list of numpy array of indexes. If verb/noun 3 is contained in actions 2,8,19,
            then output[3] will be np.array([2,8,19])
    """
    vi = []
    for v in range(actions[mode].max()+1):
        vals = actions[actions[mode] == v].index.values
        if len(vals) > 0:
            vi.append(vals)
        else:
            vi.append(np.array([0]))
    return vi


def marginalize(probs, indexes):
    mprobs = []
    for ilist in indexes:
        mprobs.append(probs[:, ilist].sum(1))
    return np.array(mprobs).T


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    xx = x
    x = x.reshape((-1, x.shape[-1]))
    e_x = np.exp(x - np.max(x, 1).reshape(-1, 1))
    res = e_x / e_x.sum(axis=1).reshape(-1, 1)
    return res.reshape(xx.shape)


def topk_recall(scores, labels, k=5, classes=None):
    unique = np.unique(labels)
    if classes is None:
        classes = unique
    else:
        classes = np.intersect1d(classes, unique)
    recalls = 0
    #np.zeros((scores.shape[0], scores.shape[1]))
    for c in classes:
        recalls += topk_accuracy(scores, labels, ks=(k,), selected_class=c)[0]
    return recalls/len(classes)


def topk_recall_multiple_timesteps(preds, labels, k=5, classes=None):
    accs = np.array([topk_recall(preds[:, t, :], labels, k, classes)
                     for t in range(preds.shape[1])])
    return accs.reshape(1, -1)


def tta(scores, labels):
    """Implementation of time to action curve"""
    rankings = scores.argsort()[..., ::-1]
    comparisons = rankings == labels.reshape(rankings.shape[0], 1, 1)
    cum_comparisons = np.cumsum(comparisons, 2)
    cum_comparisons = np.concatenate([cum_comparisons, np.ones(
        (cum_comparisons.shape[0], 1, cum_comparisons.shape[2]))], 1)
    time_stamps = np.array([2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.25, 0])
    return np.nanmean(time_stamps[np.argmax(cum_comparisons, 1)], 0)[4]


def predictions_to_json(verb_scores, noun_scores, action_scores, action_ids, a_to_vn, top_actions=100):
    """Save verb, noun and action predictions to json for submitting them to the EPIC-Kitchens leaderboard"""
    predictions = {'version': '0.1',
                   'challenge': 'action_anticipation', 'results': {}}

    row_idxs = np.argsort(action_scores)[:, ::-1]
    top_100_idxs = row_idxs[:, :top_actions]

    action_scores = action_scores[np.arange(
        len(action_scores)).reshape(-1, 1), top_100_idxs]

    for i, v, n, a, ai in zip(action_ids, verb_scores, noun_scores, action_scores, top_100_idxs):
        predictions['results'][str(i)] = {}
        predictions['results'][str(i)]['verb'] = {str(
            ii): float(vv) for ii, vv in enumerate(v)}
        predictions['results'][str(i)]['noun'] = {str(
            ii): float(nn) for ii, nn in enumerate(n)}
        predictions['results'][str(i)]['action'] = {
            "%d,%d" % a_to_vn[ii]: float(aa) for ii, aa in zip(ai, a)}
    return predictions


def upsample_to30fps(videos_path, destination_folder):
    """
    from a list of videos in a folder generate a list of images corresponing to frames at 30 fps
    :param video_path: like "/home/2/2014/nagostin/Desktop/video/"
    :param destination_folder: like "/home/2/2014/nagostin/Desktop/frames/"
    :return:
    """
    for video in os.listdir(videos_path):
        if video.endswith(".mp4"):

            video = video.split(".")[0]
            if not os.path.exists(destination_folder+video):
                os.makedirs(destination_folder+video)


            os.system('ffmpeg -i /home/2/2014/nagostin/Desktop/video/{0}.mp4 -vf "scale=-1:256,fps=30" -qscale:v 2 /home/2/2014/nagostin/Desktop/frames/{0}/{0}_frame_%010d.jpg'.format(video))

def loadNPY(file="/Volumes/Bella_li/objs/OP01-R01-PastaSalad_detections.npy"):
    """
    load npy object extracted and show in images from files computed as shown here https://github.com/fpv-iplab/rulstm/blob/master/FasterRCNN/tools/detect_video.py
    :return:
    """
    #load csv label objects into dict int_noun

    df_csv = pandas.read_csv('/Users/nicolago/Desktop/EPIC_noun_classes.csv')
    #print(df_csv[1])

    start_from = 25890

    objs = np.load(file, allow_pickle=True)
    #print(objs[1])
    count = 0

    vid = file.split("/")[-1].split("_")[0]
    print(vid)
    i=0
    for frames in os.listdir("/Volumes/Bella_li/frames/"+vid):

            if frames.endswith(".jpg"):

                #print(objs[count])
                if i > start_from:
                    image = cv2.imread("/Volumes/Bella_li/frames/" + vid + "/" + frames)
                    for n_obj in objs[count]:
                        if n_obj[5]>0.70:  # if the confidence score is quite high
                            #print(int(n_obj[1]))
                            image = cv2.rectangle(image, (int(n_obj[1]), int(n_obj[2])), (int(n_obj[3]), int(n_obj[4])), (255, 0, 0), 2)
                            cv2.putText(image, df_csv.iat[int(n_obj[0]), 1], (int(n_obj[1]), int(n_obj[2]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
                    cv2.imshow("framez", image)

                else:
                    i += 1

                count += 1
                k = cv2.waitKey(0)
                if k == 27:  # Esc key to stop
                    break

                if k == 115:
                    cv2.imwrite("bounding_boxes_example.jpg", image)

                elif k == 32:  # a key to go on
                    continue









def fromframes_tovideo(frames_path):
    """
    this function is opposite of upsample_to30fps because it sticks together frames to get a video
    :param frames_path: "/home/2/2014/nagostin/Desktop/frames/"
    :return:
    """
    for folder in os.listdir(frames_path):
        for frame in os.listdir(frames_path + folder):
            if frame.endswith(".jpg"):
                os.system("ffmpeg -f image2 -r 30 -i /home/2/2014/nagostin/Desktop/frames/{0}/{0}_frame_%010d.jpg -vcodec mpeg4 -y /home/2/2014/nagostin/Desktop/video/{0}.mp4".format(folder))



def split_train_val_test_handMask(path_to_folder):
    """
    split train and validation and test for hand mask segmenter giving the folder of hand masks as https://www.dropbox.com/s/ysi2jv8qr9xvzli/hand14k.zip?dl=0  EGTEA GAZE + DATASET
    :return:
    """

    classes = ["Images", "Masks"]
    frames = os.path.join(path_to_folder, "Images")
    masks = os.path.join(path_to_folder, "Masks")


    os.makedirs(path_to_folder + 'Frames/train')
    os.makedirs(path_to_folder + 'Frames/val')
    os.makedirs(path_to_folder + 'Frames/test')

    os.makedirs(path_to_folder + 'Maschere/train')
    os.makedirs(path_to_folder + 'Maschere/val')
    os.makedirs(path_to_folder + 'Maschere/test')

    allFileNames = os.listdir(path_to_folder+"Images")
    np.random.shuffle(allFileNames)

    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                              [int(len(allFileNames) * (0.7)),  # train
                                                               int(len(allFileNames) * (0.85))])  # validation

    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))

    ### FOR IMAGES ###
    train_FileNames = [[path_to_folder+"Images" + '/' + name, path_to_folder+"Masks" + '/' + name] for name in train_FileNames.tolist()]
    val_FileNames = [[path_to_folder+"Images" + '/' + name, path_to_folder+"Masks" + '/' + name] for name in val_FileNames.tolist()]
    test_FileNames = [[path_to_folder+"Images" + '/' + name, path_to_folder+"Masks" + '/' + name] for name in test_FileNames.tolist()]

    for name in tqdm.tqdm(train_FileNames):
        shutil.copy(name[0], path_to_folder + 'Frames/train')

    for name in tqdm.tqdm(val_FileNames):
        shutil.copy(name[0], path_to_folder + 'Frames/val')

    for name in tqdm.tqdm(test_FileNames):
        shutil.copy(name[0], path_to_folder + 'Frames/test')

    ### FOR MASKS ###



    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))

    for name in tqdm.tqdm(train_FileNames):
        n = name
        shutil.copy(n[1].split(".")[0]+".png", path_to_folder + 'Maschere/train/'+n[1].split("/")[-1].split(".")[0]+".jpg")

    for name in tqdm.tqdm(val_FileNames):
        n = name
        shutil.copy(n[1].split(".")[0]+".png", path_to_folder + 'Maschere/val/'+n[1].split("/")[-1].split(".")[0]+".jpg")

    for name in tqdm.tqdm(test_FileNames):
        n = name
        shutil.copy(n[1].split(".")[0]+".png", path_to_folder + 'Maschere/test/'+n[1].split("/")[-1].split(".")[0]+".jpg")



def split_frames_objDect(path_of_buckets):
    """
    split train and test set for detectron training
    :param path_of_buckets:
    :return:
    """
    n, p = 1, .1  # number of trials, probability of each trial
    random = np.random.binomial(n, p, 1298)
    validation=[]
    train=[]
    with open("train.txt", "w+") as train_file:
        with open("val.txt", "w+") as val_file:
            for i,folder in enumerate(os.listdir(path_of_buckets)):
                print(folder)
                if ".DS_Store" in folder:
                    continue
                if random[i-1] == 1:
                    validation.append(folder)
                    #val_file.write(folder+"\n")
                else:
                    train.append(folder)
                    #train_file.write(folder+"\n")

            for files in os.listdir("/Users/nicolago/Desktop/voc2coco-master/sample/Annotations"):

                if files.split("_")[0] in validation:
                    val_file.write(files.split(".")[0] + "\n")
                if files.split("_")[0] in train:
                    train_file.write(files.split(".")[0] + "\n")

    #print(len(validation))


def csv_to_txt(file = "/Volumes/Bella_li/labelz.txt"):
    """ parse the files in label352 noun .csv into a labels.txt file for voc2coco
    :param file:
    :return:
    """
    with open("labels.txt", "w+") as val_file:
        with open(file) as fp:
            line = fp.readline()
            while line:
                label = line.split(" ")[0]
                val_file.write(label + "\n")
                line = fp.readline()

def split_train_val_detectron(train = "/Volumes/Bella_li/train.txt", val = "/Volumes/Bella_li/val.txt"):
    """ split the dataset into train and val defined in txt files for detectron
    :return:
    """
    os.makedirs("/Volumes/Bella_li/train")
    os.makedirs("/Volumes/Bella_li/val")
    with open(train) as fp:
        line = fp.readline().rstrip()
        while line:
            shutil.copy("/Users/nicolago/Desktop/FRAMES/"+line + ".jpg",
                        "/Volumes/Bella_li/train/"+line+".jpg")
            line = fp.readline().rstrip()
    with open(val) as fp:
        line = fp.readline().rstrip()
        while line:
            shutil.copy("/Users/nicolago/Desktop/FRAMES/"+line + ".jpg",
                        "/Volumes/Bella_li/val/"+line+".jpg")
            line = fp.readline().rstrip()



def drawBBox(file="/Users/nicolanico/Desktop/data"):
    """
    load npy object extracted and show in images for json output from validation of detectron fine tuning on EGTEA gaze + obj detector
    :return:
    """
    #load csv label objects into dict int_noun

    df_csv = pandas.read_csv('/Users/nicolanico/Desktop/EPIC_noun_classes.csv')

    with open('/Users/nicolanico/Desktop/bbox_coco_2014_val_results.json') as json_file:
        with open('/Users/nicolanico/Desktop/data/annotations/instances_val2014.json') as val_data:
            data = json.load(json_file)
            images = json.load(val_data)
            for el in data:
                for obj in images["images"]:
                    if obj["id"] == el["image_id"]:
                        id = obj["id"]
                        name = obj["file_name"]
                        #print(name)
                        #print(id)

                image = cv2.imread("/Users/nicolanico/Desktop/data/val/" + name)
                for bb in data:
                    if bb["image_id"] == id:
                        if bb["score"] >0.6:
                            #print(int(bb["bbox"][0]))
                            image = cv2.rectangle(image, (int(bb["bbox"][0]), int(bb["bbox"][1])), (int(bb["bbox"][2]+bb["bbox"][0]), int(bb["bbox"][3]+bb["bbox"][1])),
                                                  (255, 0, 0), 2)
                            cv2.putText(image, df_csv.iat[bb["category_id"]-1, 1], (int(bb["bbox"][0]), int(bb["bbox"][1] + 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
                cv2.imshow(name, image)
                k = cv2.waitKey(0)
                if k == 27:  # Esc key to stop
                    break
                elif k == 32:  # a key to go on
                    continue

np.set_printoptions(threshold=sys.maxsize)  # to print all np array

def correct_HM(path_of_folder):
    """
    this function will correct hand mask in order to get only two hands: it deletes connected components grater
    than a certain threshold and consider at most two biggest cc
    :param path_of_folder : the path of the images mask predictions folder
    :return:
    """

    #here insert loop on each folder
    for folder in os.listdir(path_of_folder):
        if not ".DS_Store" in folder:

            image_files = [f for f in sorted(glob.glob(path_of_folder+folder+'/*.png'))]  # the outputs are in PGN format sorted by num frame
            print(path_of_folder+folder)
            name = image_files[0]
            vid_id = name.split("/")[-1].split("_")[0]

            objs = np.load("/home/2/2014/nagostin/Desktop/video/" + vid_id + "_detections.npy", allow_pickle=True)
            allboxes=[]
            #here insert lool on each image file
            for image in tqdm.tqdm(image_files):

                new_features = np.zeros((2, 54))  # size 2 X number of objects
                n_frame = image.split("/")[-1].split("_")[-1].split(".")[0]  # number of the frame considered
                #print(n_frame)
                img = cv2.imread(image, 0)
                #print(img)
                #img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
                num_labels, labels_im = cv2.connectedComponents(img)



                #print(im_bw)
                #for image in tqdm.tqdm(image_files):
                #print(labels_im.shape)

                #print(len(unique))
                #print(counts)
                to_delete = []
                if num_labels > 1:

                    h, w = img.shape
                    num_components, components = cv2.connectedComponents(img)
                    comp_th = 200
                    comp_list = []
                    for i in range(1, 3):

                        # Retrieve i-th component
                        comp_i = np.where(components == i, np.ones_like(img), np.zeros_like(img))

                        #print(comp_i.sum())
                    # Filter the component, if needed
                        if comp_i.sum() >= comp_th:
                            comp_list += [comp_i]

                    hand_center = []
                    for comp in comp_list:

                    # Compute hand region
                        #print(comp)
                        idxs = np.where(comp == 1)  # idxs shape [n_points, 2] the index where there is a hand rows and cols
                        #print(idxs)
                        #print(idxs)
                        temp1 = comp[idxs[0].max(axis=0)]  # max row
                        #print(temp1.shape)
                        tempidx1 = np.where(temp1 == 1)  # the index col of the max col
                        #print(tempidx1)
                        p1 = np.array([tempidx1[0].min(axis=0), idxs[0].max(axis=0)])  # punto basso a sx
                        p2 = np.array([tempidx1[0].max(axis=0), idxs[0].max(axis=0)])  # punto basso a dx
                        #y_min = idxs.min(axis=0)
                        p3 = np.array([idxs[1].min(axis=0), idxs[0].min(axis=0)])  # punto alto


                        hand_center.append([p3,p1,p2])

                    bar = []  # the weighted center of the hands

                    weight = 3

                    for hand in hand_center:

                        sum_x = hand[0][0] * weight + hand[1][0] + hand[2][0]
                        x = int(np.floor(sum_x / (2 + weight)))

                        sum_y = hand[0][1] * weight + hand[1][1] + hand[2][1]
                        y = int(np.floor(sum_y / (2 + weight)))

                        bar.append((x, y))



                    for n_obj in objs[int(n_frame)-1]:  # compute center of object
                        if n_obj[5] > 0.50:  # if the confidence score is quite high
                            # print(int(n_obj[1]))

                            #print(n_obj)
                            center_obj_x = np.floor((n_obj[1]+n_obj[3])/2)
                            center_obj_y = np.floor((n_obj[2]+n_obj[4])/2)
                            center_obj = (center_obj_x,center_obj_y)

                            for i,con in enumerate(bar):  # append its distance from center of hand to array of new features
                                if con != (0,0):
                                    distance_obj_hand = distance.euclidean(np.asarray(center_obj), np.asarray(bar[i]))/435  #la diagonale dell'immagine = 435
                                    if new_features[i][int(n_obj[0])] != 0:  # if more that 1 obj of the same type then compute the sum of their distances
                                        new_features[i][int(n_obj[0])] += (1 - distance_obj_hand)

                                    else:
                                        new_features[i][int(n_obj[0])] = 1-distance_obj_hand


                    #imshow_components(labels_im, bar)
                #print(new_features)

                new_features = np.sum(new_features, axis=0)  # get a (352,) sum vector
                #print(new_features.shape)

                allboxes.append(new_features)


            # out from all images loop but inside all folder loop do
            np.save("/home/2/2014/nagostin/Desktop/newfeat/"+vid_id+'_newfeat', allboxes)







def imshow_components(labels, bar):
    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    for con in bar:
        label_hue[con[1]][con[0]] = 0
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    cv2.imshow('labeled.png', labeled_img)
    cv2.waitKey()


def show_8_Images(Frames = "/Volumes/Bella_li/frames/OP01-R01-PastaSalad/OP01-R01-PastaSalad_frame_{:010d}.jpg"):
    """
    only shows 8 frames in a single image for thesis
    :param Frames:
    :return:
    """
    w = 100
    h = 100
    fig = plt.figure(figsize=(8, 8))
    columns = 3
    rows = 3

    for i in range(1, columns * rows + 1):
        img = plt.imread(Frames.format((int((432970/1000)-(3.5))*30)+i*30))
        fig.add_subplot(rows, columns, i)
        plt.axis('off')

        plt.imshow(img)

    plt.savefig("8images.png")

def plot_frequency_actions():
    """
    plot frequency of action verb noun depending on "lines.split(" ")[x:]:"
    :return:
    """
    path_of_txt = ["/Users/nicolago/Desktop/action_annotation/train_split1.txt",
                   "/Users/nicolago/Desktop/action_annotation/train_split2.txt",
                   "/Users/nicolago/Desktop/action_annotation/train_split3.txt",
                   "/Users/nicolago/Desktop/action_annotation/test_split1.txt",
                   "/Users/nicolago/Desktop/action_annotation/test_split2.txt",
                   "/Users/nicolago/Desktop/action_annotation/test_split3.txt"]
    n_labels = np.zeros(53)
    for el in path_of_txt:
        with open(el, "r") as sample_file:

            for lines in sample_file:
                lines = lines.strip()
                for i in np.arange(1,54):

                    if str(i) in lines.split(" ")[3:]:
                        n_labels[i-1] += 1
    print(n_labels)
    plt.bar(np.arange(1, 54), n_labels, width=0.5)
    """
    for i in np.arange(1,54):
        plt.text(i, 2, i, color='black',
                ha='center', va='top', rotation=0, fontsize=10)
    plt.xticks([])
    """
    plt.ylabel('frequency')
    plt.savefig("noun_freq.jpg")



def split_val_test(path_of_test = "/Users/nicolago/Desktop/action_annotation/test_split1.txt"):
    with open(path_of_test, 'r') as f:
        with open("val1.csv", 'w+', newline='') as val:
            with open("test1.csv", 'w+', newline='') as test:
                writer_val = csv.writer(val)
                writer_test = csv.writer(test)
                count = 0
                idx_val = 0
                idx_test = 0
                for line in f:
                    if count%2 == 0:
                        values = re.split("-| ", line)
                        v_name = values[0] + "-" + values[1] + "-" + values[2]
                        # start_time = values[3]  # Not used
                        # end_time = values[4]  # Not used
                        start_sec = str(int(values[3]) / 1000)

                        end_sec = str(int(values[4]) / 1000)

                        action_id = int(values[7]) - 1



                        writer_val.writerow([idx_val, v_name, start_sec, end_sec, action_id])
                        idx_val += 1
                    else:
                        values = re.split("-| ", line)
                        v_name = values[0] + "-" + values[1] + "-" + values[2]
                        # start_time = values[3]  # Not used
                        # end_time = values[4]  # Not used
                        start_sec = str(int(values[3]) / 1000)

                        end_sec = str(int(values[4]) / 1000)

                        action_id = int(values[7]) - 1

                        writer_test.writerow([idx_test, v_name, start_sec, end_sec, action_id])
                        idx_test += 1
                    count += 1






