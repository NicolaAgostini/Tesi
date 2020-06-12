import numpy as np
import os
import cv2
import pandas
import shutil
import random
import tqdm


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
    from a list of videos in a folder generate a list of images corresponing to frames
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

def loadNPY(file="/Volumes/Bella_li/featureobj/P26-R05-Cheeseburger_detections.npy"):
    """
    load npy object extracted and show in images
    :return:
    """
    #load csv label objects into dict int_noun

    df_csv = pandas.read_csv('/Users/nicolago/Desktop/EPIC_noun_classes.csv')
    #print(df_csv[1])

    start_from = 10000

    objs = np.load(file, allow_pickle=True)
    #print(objs[1])
    count = 0
    for vid in os.listdir("/Volumes/Bella_li/frames/"):
        if not vid.endswith(".DS_Store"):
            #vid = vid.split(".")[0]
            print(vid)
            i=0
            for frames in os.listdir("/Volumes/Bella_li/frames/"+vid):

                    if frames.endswith(".jpg"):
                        image = cv2.imread("/Volumes/Bella_li/frames/"+vid+"/"+frames)
                        #print(objs[count])
                        if i > start_from:
                            for n_obj in objs[count]:
                                if n_obj[5]>0.30:  # if the confidence score is quite high
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

    :param frames_path: "/home/2/2014/nagostin/Desktop/frames/"
    :param video_path_30fps: "/home/2/2014/nagostin/Desktop/video/"
    :return:
    """
    for folder in os.listdir(frames_path):
        for frame in os.listdir(frames_path + folder):
            if frame.endswith(".jpg"):
                os.system("ffmpeg -f image2 -r 30 -i /home/2/2014/nagostin/Desktop/frames/{0}/{0}_frame_%010d.jpg -vcodec mpeg4 -y /home/2/2014/nagostin/Desktop/video/{0}.mp4".format(folder))



def split_train_val_test_handMask(path_to_folder):
    """
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
