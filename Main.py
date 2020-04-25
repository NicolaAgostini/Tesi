from Preprocessing_bdd100k import *
from test import *
from TFLoad_bdd100k import *
from Glove import Glove
import numpy as np
from Dataset import *
from Model import *
from torch.utils.data import DataLoader



alpha = 0.2

path_to_lmdb = ["/Volumes/Bella_li/egtea/TSN-C_3_egtea_action_CE_s1_flow_model_best_fcfull_hd/",
                    "/Volumes/Bella_li/egtea/TSN-C_3_egtea_action_CE_s1_rgb_model_best_fcfull_hd/"]  # the folders that contain the .mdb files

groundtruth_path_train = "/volumes/Bella_li/egtea/action_annotation/train_split1.txt"

groundtruth_path_csv_train = "/volumes/Bella_li/egtea/training.csv"
groundtruth_path_csv_val = "/volumes/Bella_li/egtea/validation.csv"

input_dim = [1024, 1024, 352]


def get_dataset(ground_truth, batch_size, num_workers):
    """
    :param batch_size:
    :param num_workers:
    :return: the pytorch object DataLoader to handle the stream of data in dataset
    """
    a = Dataset(path_to_lmdb, ground_truth)
    return DataLoader(a, batch_size=batch_size, num_workers=num_workers,
                      pin_memory=True, shuffle= True)  # suffle true for training

def get_model():
    rgb_model = LSTMROLLING(input_dim[0], 1024)
    flow_model = LSTMROLLING(input_dim[1], 1024)
    obj_model = LSTMROLLING(input_dim[2], 352)

    model = RLSTMFusion([rgb_model, flow_model, obj_model], 1024)  # the model is the fusion of the three branches

    return model


def main():
    data_loader_train = get_dataset(groundtruth_path_csv_train, 4, 4)  # loader for training
    data_loader_val = get_dataset(groundtruth_path_csv_val, 4, 4)  # loader for validation

    model = get_model()


    #inspect_lmdb(path_to_lmdb[1])
    #txt_to_csv(groundtruth_path)


"""
def train_val(model, loaders, optimizer, epochs):
    for epoch in range(epochs):
        # define training and validation meters

        for mode in ['training', 'validation']:
            # enable gradients only if training
            with torch.set_grad_enabled(mode == 'training'):
                if mode == 'training':
                    model.train()
                else:
                    model.eval()

                for i, batch in enumerate(loaders[mode]):
                    x = batch['past_features']

                    if type(x) == list:
                        x = [xx.to(device) for xx in x]
                    else:
                        x = x.to(device)

                    y = batch['label'].to(device)

                    bs = y.shape[0]  # batch size

                    preds = model(x)

                    # take only last S_ant predictions
                    preds = preds[:, -args.S_ant:, :].contiguous()

                    # linearize predictions
                    linear_preds = preds.view(-1, preds.shape[-1])
                    # replicate the labels across timesteps and linearize
                    linear_labels = y.view(-1, 1).expand(-1,
                                                         preds.shape[1]).contiguous().view(-1)

                    loss = F.cross_entropy(linear_preds, linear_labels)
                    # get the predictions for anticipation time = 1s (index -4) (anticipation)
                    # or for the last time-step (100%) (early recognition)
                    # top5 accuracy at 1s
                    idx = -4 if args.task == 'anticipation' else -1
                    # use top-5 for anticipation and top-1 for early recognition
                    k = 5 if args.task == 'anticipation' else 1
                    acc = topk_accuracy(
                        preds[:, idx, :].detach().cpu().numpy(), y.detach().cpu().numpy(), (k,))[0] * 100

                    # store the values in the meters to keep incremental averages
                    loss_meter[mode].add(loss.item(), bs)
                    accuracy_meter[mode].add(acc, bs)

                    # if in training mode
                    if mode == 'training':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # compute decimal epoch for logging
                    e = epoch + i / len(loaders[mode])

                    # log training during loop
                    # avoid logging the very first batch. It can be biased.
                    if mode == 'training' and i != 0 and i % args.display_every == 0:
                        log(mode, e, loss_meter[mode], accuracy_meter[mode])

                # log at the end of each epoch
                log(mode, epoch + 1, loss_meter[mode], accuracy_meter[mode],
                    max(accuracy_meter[mode].value(), best_perf) if mode == 'validation'
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
