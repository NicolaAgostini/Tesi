import torch
import os
from torch import nn
from pretrainedmodels import bninception
from torchvision import transforms
from glob import glob
from PIL import Image
import lmdb
from tqdm import tqdm
from os.path import basename
from argparse import ArgumentParser


env = lmdb.open("/aulahomes2/2/2014/nagostin/Desktop/RGB_Gaze", map_size=1099511627776, lock=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = bninception(pretrained=None)
state_dict = torch.load("/aulahomes2/2/2014/nagostin/Desktop/tsn-pytorch/egteabninception__rgb_model_best.pth.tar")['state_dict']
state_dict = {k.replace('module.base_model.','') : v for k,v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)


model.last_linear = nn.Identity()
model.global_pool = nn.AdaptiveAvgPool2d(1)

model.to(device)

transform = transforms.Compose([
    transforms.Resize([256, 454]),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[[2,1,0],...]*255), #to BGR
    transforms.Normalize(mean=[104, 117, 128],
                         std=[1, 1, 1]),
])

imgs = sorted(glob('data/sample_rgb/*.jpg'))

model.eval()
for file in os.listdir("/aulahomes2/2/2014/nagostin/Desktop/frames/"):
    print(file)
    video_name = file.split("_")[0] + '_frame_{:010d}.jpg'
    for i,im in enumerate(tqdm(sorted(os.listdir("/aulahomes2/2/2014/nagostin/Desktop/frames/"+ file+"/")))):
        key = video_name.format(i+1)
        img = Image.open(im)
        data = transform(img).unsqueeze(0).to(device)
        feat = model(data).squeeze().detach().cpu().numpy()
        with env.begin(write=True) as txn:
            txn.put(key.encode(),feat.tobytes())