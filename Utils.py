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

    df_csv = pandas.read_csv('/Users/nicolanico/Desktop/EPIC_noun_classes.csv')
    #print(df_csv[1])

    start_from = 12500

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
    :param path_of_buckets:
    :return:
    """
    n, p = 1, .2  # number of trials, probability of each trial
    random = np.random.binomial(n, p, 86)
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
            for files in os.listdir("/Volumes/Bella_li/Frames_da_labellare"):
                if files.split("_")[0] in validation:
                    val_file.write(files.split(".")[0] + "\n")
                if files.split("_")[0] in train:
                    train_file.write(files.split(".")[0] + "\n")

    #print(len(validation))


def csv_to_txt(file = "labelz.txt"):
    """ parse the files in label352 noun .csv into a labels.txt file for voc2coco
    :param file:
    :return:
    """
    with open("labels.txt", "w+") as val_file:
        with open(file) as fp:
            line = fp.readline()
            while line:
                label = line.split(",")[1]
                val_file.write(label + "\n")
                line = fp.readline()

def split_train_val_detectron(train = "/Volumes/Bella_li/train.txt", val = "/Volumes/Bella_li/val.txt"):
    """ split the dataset into train and val defined in txt files
    :return:
    """
    #os.makedirs("/Volumes/Bella_li/train")
    #os.makedirs("/Volumes/Bella_li/val")
    with open(train) as fp:
        line = fp.readline().rstrip()
        while line:
            shutil.copy("/Volumes/Bella_li/Frames_da_labellare/"+line+ ".jpg",
                        "/Volumes/Bella_li/train/"+line+".jpg")
            line = fp.readline().rstrip()
    with open(val) as fp:
        line = fp.readline().rstrip()
        while line:
            shutil.copy("/Volumes/Bella_li/Frames_da_labellare/"+line+ ".jpg",
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
            print(folder)
            name = image_files[0]
            vid_id = name.split("/")[-1].split("_")[0]

            objs = np.load("/Volumes/Bella_li/objs/" + vid_id + "_detections.npy", allow_pickle=True)
            allboxes=[]
            #here insert lool on each image file
            for image in tqdm.tqdm(image_files):

                new_features = np.zeros((2, 352))
                n_frame = image.split("/")[-1].split("_")[-1].split(".")[0]  # number of the frame considered
                #print(n_frame)
                img = cv2.imread(image, 0)
                #print(img)
                #img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
                num_labels, labels_im = cv2.connectedComponents(img)



                #print(im_bw)
                #for image in tqdm.tqdm(image_files):
                #print(labels_im.shape)
                unique, counts = np.unique(labels_im, return_counts=True)
                #print(len(unique))
                print(counts)
                to_delete = []
                if len(unique) > 1:
                    for i,el in enumerate(counts):

                        if el<200:  # threshold for hand
                            to_delete.append(i)


                    for i,row in enumerate(labels_im):
                        for j,item in enumerate(row):
                            if item > 2 or item in to_delete:
                                labels_im[i][j] = 0  # consider only 2 hands because the index of cc are given following decreasing cardinality

                #now find the 3 vertex of every cc

                    h = [(0, 0),(0, 0)]
                    lsx = [(0, 0),(0, 0)]
                    ldx = [(0, 0),(0, 0)]

                    last_row = labels_im.shape[0]

                    for con in[1,2]:
                        for i, row in enumerate(labels_im):
                            for j, el in enumerate(row):
                                if el == con and h[con-1] == (0, 0):
                                    h[con-1] = (j,i)
                                if (el == con and i == last_row-1) or (el == con and labels_im[i][j-1] !=con):
                                    lsx[con - 1] = (j, i)
                                    if i == last_row-1:
                                        break
                                if (el == con and labels_im[i][j+1] != con):
                                    ldx[con - 1] = (j, i)

                    print(h, lsx, ldx)

                    # compute weighted centroid

                    bar=[(0,0),(0,0)]

                    weight = 3

                    for con in [0, 1]:
                        sum_x=h[con][0]*weight + lsx[con][0] + ldx[con][0]
                        x = int(np.floor(sum_x/(2+weight)))

                        sum_y = h[con][1] * weight + lsx[con][1] + ldx[con][1]
                        y = int(np.floor(sum_y / (2 + weight)))

                        bar[con] = (x,y)
                    print(bar)



                    for n_obj in objs[int(n_frame)]:  # compute center of object
                        if n_obj[5] > 0.60:  # if the confidence score is quite high
                            # print(int(n_obj[1]))
                            #print(n_obj)

                            center_obj_x = np.floor((n_obj[1]+n_obj[3])/2)
                            center_obj_y = np.floor((n_obj[2]+n_obj[4])/2)
                            center_obj = (center_obj_x,center_obj_y)

                            for i,con in enumerate(bar):  # append its distance from center of hand to array of new features
                                if con != (0,0):
                                    distance_obj_hand = distance.euclidean(np.asarray(center_obj), np.asarray(bar[i]))/435  #la diagonale dell'immagine
                                    if new_features[i][int(n_obj[0])] != 0:
                                        if 1 - distance_obj_hand > new_features[i][int(n_obj[0])]:
                                            new_features[i][int(n_obj[0])] = 1 - distance_obj_hand

                                    else:
                                        new_features[i][int(n_obj[0])] = 1-distance_obj_hand

                #print(new_features)
                    #imshow_components(labels_im, bar)
                new_features = np.mean(new_features, axis=0)  # get a (352,) mean vector
                #print(new_features.shape)

            allboxes.append(new_features)


            # out from all images loop but insiede all folder loop do
            np.save("/Volumes/Bella_li/newfeat/"+name+'_newfeat', allboxes)







def imshow_components(labels, bar):
    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    for con in [0, 1]:
        label_hue[bar[con][1]][bar[con][0]] = 0
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    cv2.imshow('labeled.png', labeled_img)
    cv2.waitKey()