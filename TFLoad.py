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
    image = open(path1, 'rb').read()
    n_frame = os.path.splitext(os.path.split(path1)[-1])[0].split("-")[-1]
    name = os.path.split(path1)[0].split("/")[-1]
    label = info[name]
    #print(label, n_frame)
    l = 1
    if label != "No" and int(label) == int(n_frame):
        #print("ok")
        l = 0

    return image, l


def _convert_to_example(image_buffer, label):
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': _int64_feature(label),
        'encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
    return example

def load_dataset():
    """
    :return: writes if .tfrecords all the frames with label and name
    """
    #tf.print(dataset, output_stream=sys.stderr)
    all_images = []

    for img_d in os.listdir("/Users/nicolago/Desktop/test/trainimg/"):
        if not img_d.startswith('.'):
            for i in range(16):
                all_images.append("/Users/nicolago/Desktop/test/trainimg/"+img_d+"/"+img_d+"-"+str(i+1)+".jpg")

    #print(all_images)

    with open("/Users/nicolago/Desktop/test/traingroundtruth.json", 'r') as f:
        info = json.load(f)

        with tf.io.TFRecordWriter("/Users/nicolago/Desktop/test/img.tfrecords") as writer:
            for filename in tqdm(all_images):
                image_buffer, label = _process_image(filename, info)
                example = _convert_to_example(image_buffer, label)
                writer.write(example.SerializeToString())




def cose(path):

    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(read_tfrecord)
    #print(tf.print(dataset))
    dataset = dataset.shuffle(1000 + 3 * 1)
    dataset = dataset.batch(1)
    for image, label in dataset.take(1):
        plt.title(label.numpy())
        plt.imshow(image)


def read_tfrecord(path_of_file):
    image_dataset = tf.data.TFRecordDataset(path_of_file)
    IMG_SIZE = 112
    # Create a dictionary describing the features.
    image_feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'encoded': tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        feature = tf.io.parse_single_example(example_proto, image_feature_description)

        image = feature['encoded']
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [224, 224])
        image /= 255.0  # normalize to [0,1] range



        return image, feature['label']

    dataset = image_dataset.map(_parse_image_function)

    BATCH_SIZE = 32

    for image, label in dataset.take(3):
        print(label)
        plt.imshow(image)
        plt.show()




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






