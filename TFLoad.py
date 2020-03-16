from __future__ import with_statement
import tensorflow as tf
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import tensorflow_datasets as tfds
from PIL import Image

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import datetime as dt
import sys
from tqdm import tqdm
import cv2
import glob







def convert_toarray(path):
    """
    :param path: path of the images root folder
    :return: the array of images of the videos and labels for every video
    """
    train_x = []
    train_y = []
    for folder in os.listdir(path):
        if not folder.startswith('.'):
            y = []
            for img in os.listdir(path+folder):
                if not img.startswith('.'):
                    i = Image.open(path+folder+"/"+img)
                    y.append(np.array(i))
            train_x.append(y)  #quindi train_x conterrà tutte i frames di un video
            train_y.append(get_label1(folder))

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_x = train_x.astype(np.float32)

    train_x /= 255.0

    print(train_x.shape, train_y.shape)

    print(train_y[0])

    return train_x, train_y







def get_label1(file_name):
    """
    :return: the label of the video frame
    """
    print(file_name)
    with open("/Users/nicolago/Desktop/test/traingroundtruth.json", 'r') as f:
        info = json.load(f)
        return str(info[file_name])





def res_net_block(input_data, filters, conv_size):
    x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, input_data])
    #x = layers.Activation('relu')(x)  # ci va o no???!!!
    return x






def load_datas():
    x_train, y_train = convert_toarray("/Users/nicolago/Desktop/test/imgs/")

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64).shuffle(1000)

    for f in train_dataset.take(5):
        print(f)



    #labeled_ds = list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def model():
    inputs = keras.Input(shape=(16, 112, 112, 3))
    x = layers.TimeDistibuted(layers.Conv2D(64, (3,3), strides=(2, 2), activation='relu')(inputs))
    x = layers.TimeDistibuted(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x))
    x = res_net_block(x, 64, 3)


    """
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(3)(x)
    num_res_net_blocks = 10
    for i in range(num_res_net_blocks):
        x = res_net_block(x, 64, 3)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    res_net_model = keras.Model(inputs, outputs)
    """



def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))






def _process_image(path1, info):  # if the current frame is the frame where it stop than label is 0 otherwise 1
    """
    :param path: path of the image
    :param info: dict of the json groundtruth
    :return:
    """

    #n_frame = os.path.splitext(os.path.split(path1)[-1])[0].split("-")[-1]
    name = path1.split("/")[-1]
    label = info[name]
    #print(label, n_frame)
    if label == "No":
        label = -1  # -1 means NO SHOW
    #name = os.path.split(path1)[-1].split(".")[0]
    return label, name


def _convert_to_example(image_buffer, filename, label):
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': _int64_feature(label),
        "filename": _bytes_feature(tf.compat.as_bytes(filename)),
        'encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
    return example

def load_dataset():
    """
    :return: writes if .tfrecords all the frames with label and name
    """
    #tf.print(dataset, output_stream=sys.stderr)
    videos = []
    with open("/Users/nicolago/Desktop/test/traingroundtruth.json", 'r') as f:
        info = json.load(f)
        for img_d in os.listdir("/Users/nicolago/Desktop/test/trainimg/"):
            temp = []
            if not img_d.startswith('.'):
                for i in range(16):
                    img = Image.open("/Users/nicolago/Desktop/test/trainimg/"+img_d+"/"+img_d+"-"+str(i+1)+".jpg")
                    temp.append(np.asarray(img, dtype='uint8'))
                temp = np.array(temp)
                with tf.io.TFRecordWriter("/Users/nicolago/Desktop/test/"+img_d+".tfrecords") as writer:
                    # Read and resize all video frames, np.uint8 of size [N,H,W,3]
                    frames = temp
                    label, name = _process_image("/Users/nicolago/Desktop/test/trainimg/"+img_d, info)
                    features = {}
                    features['num_frames'] = _int64_feature(frames.shape[0])
                    features['height'] = _int64_feature(frames.shape[1])
                    features['width'] = _int64_feature(frames.shape[2])
                    features['channels'] = _int64_feature(frames.shape[3])
                    features['class_label'] = _int64_feature(label)
                    #features['class_text'] = _bytes_feature(tf.compat.as_bytes(example['class_label']))
                    features['filename'] = _bytes_feature(tf.compat.as_bytes(name))

                    # Compress the frames using JPG and store in as bytes in:
                    # 'frames/000001', 'frames/000002', ...
                    for i in range(len(frames)):
                        ret, buffer = cv2.imencode(".jpg", frames[i])
                        features["frames/{:03d}".format(i)] = _bytes_feature(tf.compat.as_bytes(buffer.tobytes()))

                    tfrecord_example = tf.train.Example(features=tf.train.Features(feature=features))
                    writer.write(tfrecord_example.SerializeToString())








def decode(serialized_example):
    '''
    Given a serialized example in which the frames are stored as
    compressed JPG images 'frames/0001', 'frames/0002' etc., this
    function samples SEQ_NUM_FRAMES from the frame list, decodes them from
    JPG into a tensor and packs them to obtain a tensor of shape (N,H,W,3).
    Returns the the tuple (frames, class_label (tf.int64)
    :param serialized_example: serialized example from tf.data.TFRecordDataset
    :return: tuple: (frames (tf.uint8), class_label (tf.int64)
    '''

    # Prepare feature list; read encoded JPG images as bytes
    features = dict()
    features["class_label"] = tf.io.FixedLenFeature((), tf.int64)
    for i in range(16):
        features["frames/{:03d}".format(i)] = tf.io.FixedLenFeature((), tf.string)

    # Parse into tensors
    parsed_features = tf.io.parse_single_example(serialized_example, features)

    # Decode the encoded JPG images
    images = []
    for i in range(16):
        images.append(tf.image.decode_jpeg(parsed_features["frames/{:03d}".format(i)]))

    # Pack the frames into one big tensor of shape (N,H,W,3)
    images = tf.stack(images)
    label = tf.cast(parsed_features['class_label'], tf.int64)

    # Randomly sample offset ... ? Need to produce strings for dict indices after this
    # offset = tf.random_uniform(shape=(), minval=0, maxval=label, dtype=tf.int64)

    return images, label









def read_tfrecord():
    """
    :param path_of_file:
    :return: read in adataset tf image, its name (with the infor in which frame u are) and the label 0 for stop 1 for motion
    """
    NUM_EPOCHS = 1
    tfrecord_files = glob.glob("/Users/nicolago/Desktop/test/*.tfrecords")
    tfrecord_files.sort()
    dataset = tf.data.TFRecordDataset(tfrecord_files)

    dataset = dataset.repeat(NUM_EPOCHS)
    dataset = dataset.map(decode)

    for image, label in dataset.take(1):
        print(label.numpy())
        for frame in range(16):
            cv2.imshow("image", np.array(image[frame]))
            cv2.waitKey(100)
        key = cv2.waitKey(0)
        if key == ord('q'):
            exit()




class Tensors():

    def __init__(self, observed_frames):
        self.observed_frames = observed_frames

    def get_traintestnp(self):
        """
        :return: the numpy arrays for train and test
        """
        x_train, y_train = convert_toarray("volumes/HD/bdd100k/trainimg/")
        x_test, y_test = convert_toarray("volumes/HD/bdd100k/valimg/")

        return x_train, y_train, x_test, y_test






