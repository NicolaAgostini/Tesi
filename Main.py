from Preprocessing_bdd100k import *
from test import *
from TFLoad_bdd100k import *
from Glove import Glove
import numpy as np
from Dataset import *
from Model import *
from torch.utils.data import DataLoader
from torch.nn import functional as F



device = 'cuda' if torch.cuda.is_available() else 'cpu'

mode = "train"  # if train or test

alpha = 0.2

path_to_lmdb = ["/Volumes/Bella_li/egtea/TSN-C_3_egtea_action_CE_s1_flow_model_best_fcfull_hd/",
                    "/Volumes/Bella_li/egtea/TSN-C_3_egtea_action_CE_s1_rgb_model_best_fcfull_hd/"]  # the folders that contain the .mdb files

groundtruth_path_train = ["/volumes/Bella_li/egtea/action_annotation/train_split1.txt",
                          "/volumes/Bella_li/egtea/action_annotation/train_split2.txt",
                          "/volumes/Bella_li/egtea/action_annotation/train_split3.txt"]


path_to_csv_trainval = ['/Volumes/Bella_li/egtea/training1.csv', '/Volumes/Bella_li/egtea/validation1.csv']  # path of train val csv



### SOME MODEL'S VARIABLES ###

input_dim = [1024, 1024, 352]
batch_size = 1
seq_len = 14

learning_rate = 0.01
momentum = 0.9

epochs = 100



def get_dataset(ground_truth, batch_size, num_workers):
    """
    :param batch_size:
    :param num_workers:
    :return: the pytorch object DataLoader to handle the stream of data in dataset
    """
    a = Dataset(path_to_lmdb, ground_truth)
    return DataLoader(a, batch_size=batch_size, num_workers=num_workers,
                      pin_memory=True, shuffle=True)  # suffle true for training

"""
def get_model():
    rgb_model = LSTMROLLING(input_dim[0], 1024)
    flow_model = LSTMROLLING(input_dim[1], 1024)
    obj_model = LSTMROLLING(input_dim[2], 352)

    model = RLSTMFusion([rgb_model, flow_model, obj_model], 1024)  # the model is the fusion of the three branches

    return model
    
"""

def initialize_trainval_csv(which_split):
    """
    generate training and validation csv
    :param which_split: {1,2,3} the split of egtea gaze +
    :return:
    """

    list_path = generate_train_val_txt(groundtruth_path_train[which_split], str(which_split))
    path = [txt_to_csv(list_path[0], "training"+str(which_split))]
    path.append(txt_to_csv(list_path[1], "validation"+str(which_split)))
    return path


def main():



    #path = initialize_trainval_csv(1)  # to generate training and validation csv

    
    smoothed_labels = label_smmothing("prior")


    model = BaselineModel(batch_size, seq_len, input_dim)
    model = model.to(device)

    if mode == "train":
        #data_loader_train = get_dataset(path[0], 4, 4)  # loader for training
        #data_loader_val = get_dataset(path[1], 4, 4)  # loader for validation

        data_loader_train = get_mock_dataloader()
        data_loader_val = get_mock_dataloader()

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

        train_val(model, [data_loader_train, data_loader_val], optimizer, epochs, smoothed_labels)



def train_val(model, loaders, optimizer, epochs, smoothed_labels):
    """

    :param model:
    :param loaders:
    :param optimizer:
    :param epochs:
    :return:
    """
    for epoch in range(epochs):

        for mode in [0, 1]:  # 0 for training, 1 for validation
            # enable gradients only if training
            with torch.set_grad_enabled(mode == 0):
                if mode == 0:
                    model.train()
                else:
                    model.eval()

                for i, batch in enumerate(loaders[mode]):
                    x = batch['past_features']  # load in batch the next "past_features" datas of size (batch_size * 14 * 1024(352)


                    x = [xx.to(device) for xx in x]  # if input is a size (for multiple branch) then load in the devices


                    y = batch['label'].to(device)  # get the label of the batch (batch * 1)
                    bs = y.shape[0]  # batch size

                    temp = []
                    for j in range(bs):
                        temp.append(smoothed_labels[y[j]])
                    y = temp  # y size (batch, 106) and contains the smoothed labels for every y
                    y = torch.FloatTensor(y)


                    print(y.size())




                    preds = model(x)

                    # take only the last 8 predictions
                    preds = preds[:, -8:, :].contiguous()

                    # linearize predictions
                    linear_preds = preds.view(-1, preds.shape[-1])  # (batch * 8 , 106)
                    # replicate the labels across timesteps and linearize
                    print("Output " + str(linear_preds.size()))
                    #print(preds.size())
                    linear_labels = y.expand(preds.shape[1], -1).contiguous()
                    print(linear_labels.size())

                    loss = nn.BCEWithLogitsLoss()(linear_preds, linear_labels)  # loss function for smoothed labels
                    # get the predictions for anticipation time = 1s (index -4) (anticipation)
                    # or for the last time-step (100%) (early recognition)
                    # top5 accuracy at 1s
                    """
                    idx = -4 if args.task == 'anticipation' else -1
                    # use top-5 for anticipation and top-1 for early recognition
                    k = 5 if args.task == 'anticipation' else 1
                    acc = topk_accuracy(
                        preds[:, idx, :].detach().cpu().numpy(), y.detach().cpu().numpy(), (k,))[0] * 100

                    # store the values in the meters to keep incremental averages
                    loss_meter[mode].add(loss.item(), bs)
                    accuracy_meter[mode].add(acc, bs)
                    """

                    # if in training mode
                    if mode == 0:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # compute decimal epoch for logging
                    e = epoch + i / len(loaders[mode])

                    # log training during loop
                    # avoid logging the very first batch. It can be biased.
                    """
                    if mode == 0 and i != 0 and i % args.display_every == 0:
                        print(mode, e, loss_meter[mode], accuracy_meter[mode])
                    

                # log at the end of each epoch
                print(mode, epoch + 1, loss_meter[mode], accuracy_meter[mode],
                    max(accuracy_meter[mode].value(), best_perf) if mode == 1
                    else None, green=True)
                

        if best_perf < accuracy_meter['validation'].value():
            best_perf = accuracy_meter['validation'].value()
            is_best = True
        else:
            is_best = False

        # save checkpoint at the end of each train/val epoch
        save_model(model, epoch + 1, accuracy_meter['validation'].value(), best_perf,
                   is_best=is_best)
    """



def label_smmothing(set_modality="standard", alpha=0.1, temperature = 0):
    """
    :param set_modality: standard or softmax or prior
    :param alpha:
    :param temperature: provided for softmax
    :return: smoothed labels depending on the specific modality
    """
    a = Glove("/Users/nicolago/Desktop/Glove.6B/", alpha, set_modality, temperature)

    # print(a.find_similar("move")[1:6])
    b = a.get_ysoft()
    return b


if __name__ == '__main__':
    main()
