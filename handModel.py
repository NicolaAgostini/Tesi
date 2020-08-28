import os
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import torch
import numpy as np
import segmentation_models_pytorch as smp
import glob
import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE= "+device)

import cv2
import matplotlib.pyplot as plt
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


DATA_DIR = "/home/2/2014/nagostin/Desktop/hand14k/"

x_train_dir = os.path.join(DATA_DIR, 'Frames/train')
y_train_dir = os.path.join(DATA_DIR, 'Maschere/train')

x_valid_dir = os.path.join(DATA_DIR, 'Frames/val')
y_valid_dir = os.path.join(DATA_DIR, 'Maschere/val')

x_test_dir = os.path.join(DATA_DIR, 'Frames/test')
y_test_dir = os.path.join(DATA_DIR, 'Maschere/test')

def visualize(iter, folder, **images):
    """PLot images in one row."""
    n = len(images)

    for i, (name, image) in enumerate(images.items()):

        if image.shape[0]<4:
            image = np.transpose(image,(1,2,0))

        image = Image.fromarray(image.astype('uint8'))

        image.save('/home/2/2014/nagostin/Desktop/Tesi/predictions/'+folder+'/' + iter +'.png')




class Dataset(BaseDataset):
    """Hand mask egtea gaze + Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """


    def __init__(
            self,
            images_dir,
            masks_dir= None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        if masks_dir is not None:
            self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        else:
            self.masks_fps = masks_dir

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(image.shape)
        if self.masks_fps is not None:
            mask = cv2.imread(self.masks_fps[i], 0)
        #print(self.masks_fps[0])
        #print(mask)

        # apply augmentations
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']

            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            return image, mask
        else:
            # apply preprocessing
            if self.preprocessing:
                sample = self.preprocessing(image=image)
                image= sample['image']

            return image

    def __len__(self):
        return len(self.ids)

dataset = Dataset(x_train_dir, y_train_dir)


### we didin't used augmentation for training but it can be useful to improve the performance ###
def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value


def to_tensor(x, **kwargs):  # MAKE SURE IMAGES SHAPE ARE DIVISIBLE BY 32 since they will be subsampled 5 times
    if x.shape[-1] > 3:

        x = np.stack((x,) * 1, axis=-1)
        x = x/255

    npad = ((0, 0), (5, 6), (0, 0))
    x = np.pad(x, pad_width=npad, mode='constant', constant_values=0)

    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

"""
augmented_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    augmentation=get_training_augmentation()
)
"""

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['car']
ACTIVATION = 'sigmoid'  # could be None for logits or 'softmax2d' for multicalss segmentation
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# create segmentation model with pretrained encoder
model = smp.FPN(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=len(CLASSES),
    activation=ACTIVATION,
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train_dataset = Dataset(
    x_train_dir,
    y_train_dir,

    preprocessing=get_preprocessing(preprocessing_fn)
)

valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,

    preprocessing=get_preprocessing(preprocessing_fn)
)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=4)

loss = smp.utils.losses.DiceLoss()
metrics = [
    smp.utils.metrics.IoU(threshold=0.5),
]

optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=0.0001),
])

train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

max_score = 0

### TRAINING ###
"""
for i in range(0, 5):

    print('\nEpoch: {}'.format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    # do something (save model, change lr, etc.)
    if max_score < valid_logs['iou_score']:
        max_score = valid_logs['iou_score']
        torch.save(model, './best_model.pth')
        print('Model saved!')

    if i == 25:
        optimizer.param_groups[0]['lr'] = 1e-5
        print('Decrease decoder learning rate to 1e-5!')

"""


#best_model = torch.load('./best_model.pth', map_location=torch.device('cpu'))  # load best model into cpu
best_model = torch.load('./best_model.pth')  # load best model into gpu
### TEST ###  # best model giving IoU 0.80
test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    preprocessing=get_preprocessing(preprocessing_fn)
)

test_dataset_vis = Dataset(
    x_test_dir, y_test_dir,
)

def predict_folder(best_model, pathFrames = "/home/2/2014/nagostin/Desktop/frames/"):
    """
    :return:
    """
    for folder in os.listdir("/home/2/2014/nagostin/Desktop/frames/"):
        pred_dir = "/home/2/2014/nagostin/Desktop/frames/"+folder
        os.makedirs("/home/2/2014/nagostin/Desktop/Tesi/predictions/" + folder+"/")

        pred_dataset = Dataset(
            pred_dir,
            preprocessing=get_preprocessing(preprocessing_fn)
        )

        image_files = [f for f in glob.glob("/home/2/2014/nagostin/Desktop/frames/"+folder+"/*.jpg")]
        count = 0
        for filepath in tqdm.tqdm(image_files):

            image = pred_dataset[count]

            count += 1
            x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)

            pr_mask = best_model.predict(x_tensor)
            pr_mask = (pr_mask.squeeze().cpu().numpy().round())
            visualize(filepath.split("/")[-1].split(".")[0], folder,
                predicted_mask=pr_mask
            )

predict_folder(best_model)

###  show a sample of predictions  ###
"""
for i in range(10):
    n = np.random.choice(len(test_dataset))

    image_vis = test_dataset_vis[n][0].astype('uint8')
    image, gt_mask = test_dataset[n]
    print(image.shape)
    gt_mask = gt_mask.squeeze()

    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    print(x_tensor.size())
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())

    visualize(
        i,
        image=image_vis,
        ground_truth_mask=gt_mask,
        predicted_mask=pr_mask
    )
"""
"""
test_dataloader = DataLoader(test_dataset)

test_epoch = smp.utils.train.ValidEpoch(
    model=best_model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)

logs = test_epoch.run(test_dataloader)
"""




