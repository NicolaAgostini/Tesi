import torch
from test import *
from Glove import Glove
import numpy as np
from Dataset import *
from Model import *
from torch.utils.data import DataLoader
from torch.nn import functional as F
from Utils import topk_accuracy, ValueMeter, topk_accuracy_multiple_timesteps, get_marginal_indexes, marginalize, softmax,  topk_recall_multiple_timesteps, tta, predictions_to_json

root_path = "/home/2/2014/nagostin/Desktop/"


device = 'cuda' if torch.cuda.is_available() else 'cpu'



# mode = "train"  # if train or test

alpha = 0.2

path_to_lmdb = [root_path+"egtea/TSN-C_3_egtea_action_CE_s1_flow_model_best_fcfull_hd/",
                    root_path+"egtea/TSN-C_3_egtea_action_CE_s1_rgb_model_best_fcfull_hd/",
                    root_path+"egtea/obj/"]  # the folders that contain the .mdb files

### PATH OF TXT FOR TRAINING AND VALIDATION ###

groundtruth_path_train = [root_path+"egtea/action_annotation/train_split1.txt",
                          root_path+"egtea/action_annotation/train_split2.txt",
                          root_path+"egtea/action_annotation/train_split3.txt"]

groundtruth_path_test = [root_path+"egtea/action_annotation/test_split1.txt",
                          root_path+"egtea/action_annotation/test_split2.txt",
                          root_path+"egtea/action_annotation/test_split3.txt"]


###


path_to_csv_trainval = [root_path+"egtea/training1.csv", root_path+"egtea/validation1.csv"]  # path of train val csv





### SOME MODEL'S VARIABLES ###

input_dim = [1024, 1024, 352]
batch_size = 4
seq_len = 14

learning_rate = 0.001


epochs = 10

display_every = 10



def get_dataset(ground_truth, batch_size, num_workers):
    """
    :param batch_size:
    :param num_workers:
    :return: the pytorch object DataLoader to handle the stream of data in dataset
    """
    a = Dataset(path_to_lmdb, ground_truth)
    return DataLoader(a, batch_size=batch_size, num_workers=num_workers,
                      pin_memory=True, shuffle=True)  # suffle true for training


def initialize_trainval_csv(which_split):
    """
    generate training and validation csv
    :param which_split: {1,2,3} the split of egtea gaze +
    :return:
    """

    path = [txt_to_csv(groundtruth_path_train[which_split-1], "training"+str(which_split))]
    path.append(txt_to_csv(groundtruth_path_test[which_split-1], "validation"+str(which_split)))
    return path



def main():

    #path = initialize_trainval_csv(1)  # to generate training and validation csv depending on split defined by authors of egtea gaze +

    #smoothed_labels = label_smmothing("prior")  # for smoothed labels
    

    model = BaselineModel(batch_size, seq_len, input_dim)

    model.to(device)
    print(model)

    #if mode == "train":
    data_loader_train = get_dataset(path_to_csv_trainval[0], batch_size, 4)  # loader for training
    data_loader_val = get_dataset(path_to_csv_trainval[1], batch_size, 4)  # loader for validation

    #data_loader_train = get_mock_dataloader()
    #data_loader_val = get_mock_dataloader()

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    #train_val(model, [data_loader_train, data_loader_val], optimizer, epochs, smoothed_labels)  # with smoothed labels

    train_val(model, [data_loader_train, data_loader_val], optimizer, epochs)






def train_val(model, loaders, optimizer, epochs):
    """

    :param model:
    :param loaders:
    :param optimizer:
    :param epochs:
    :return:
    """
    best_perf = 0
    for epoch in range(epochs):


        loss_meter = {'0': ValueMeter(), '1': ValueMeter()}
        accuracy_meter = {'0': ValueMeter(), '1': ValueMeter()}

        for mode in [0, 1]:  # 0 for training, 1 for validation

            # enable gradients only if training
            with torch.set_grad_enabled(mode == 0):
                if mode == 0:

                    model.train()
                else:
                    model.eval()

                for i, batch in enumerate(loaders[mode]):

                    #h = tuple([e.data for e in h])

                    x = batch['past_features']  # load in batch the next "past_features" datas of size (batch_size * 14 * 1024(352)

                    x = [xx.to(device) for xx in x]  # if input is a list (for multiple branch) then load in the device gpu

                    y = batch['label'].to(device)  # get the label of the batch (batch, 1)
                    #y = batch['label']

                    """
                    ###  FOR SMOOTHED LABELS ###

                    y_temp = y  # label (batch_size, 1) to use for top-k accuracy
                    bs = y.shape[0]  # batch size
                    preds = model(x)
                    preds = preds.contiguous()

                    temp = []
                    for j in range(bs):
                        temp += 8*[smoothed_labels[y[j]]]
                    y = temp  # y size (batch*8, 106) and contains the smoothed labels for every y where 8 is 8 anticipation steps
                    y = torch.FloatTensor(y).cuda()

                    # USE WITH :

                    linear_preds = preds.view(-1, preds.shape[-1])  # (batch * 8 , 106) ogni riga ha una label corrispondente al timestamp
                    
                    linear_labels = y
                     
                    loss = nn.BCEWithLogitsLoss()(linear_preds, linear_labels)  # loss function for smoothed labels

                    #print(y[1])
                    """

                    bs = y.shape[0]  # batch size
                    #print(bs)

                    preds = model(x)
                    preds = preds.contiguous()
                    #print("output of the model " + str(preds.size()))

                    # linearize predictions
                    linear_preds = preds.view(-1, preds.shape[-1])  # (batch * 8 , 106) ogni riga ha una label corrispondente al timestamp

                    #print(linear_preds)

                    linear_labels = y.view(-1, 1).expand(-1, preds.shape[1]).contiguous().view(-1)

                    #print("labels ", linear_labels)

                    loss = F.cross_entropy(linear_preds, linear_labels)

                    #print(loss)

                    # get the predictions for anticipation time = 1s (index -4) (anticipation)
                    # or for the last time-step (100%) (early recognition)
                    # top5 accuracy at 1s
                    idx = -4

                    k = 5  # top 5 anticipation

                    acc = topk_accuracy(preds[:, idx, :].detach().cpu().numpy(), y.detach().cpu().numpy(), (k,))[0] * 100  # top 5 accuracy percentage
                    #print(acc)
                    #acc = topk_accuracy(preds[:, idx, :].detach().cpu().numpy(), y_temp.detach().cpu().numpy(), (k,))[0] * 100  # for smoothed labels

                    # store the values in the meters to keep incremental averages
                    loss_meter[str(mode)].add(loss.item(), bs)
                    accuracy_meter[str(mode)].add(acc, bs)


                    # if in training mode
                    if mode == 0:
                        optimizer.zero_grad()
                        loss.backward(retain_graph=True)
                        optimizer.step()

                    # compute decimal epoch for logging
                    e = epoch + i / len(loaders[mode])

                    # log training during loop
                    # avoid logging the very first batch. It can be biased.

                    if mode == 0 and i != 0 and i % display_every == 0:
                        log(mode, e, loss_meter[str(mode)], accuracy_meter[str(mode)])
                    

                # log at the end of each epoch
                log(mode, epoch + 1, loss_meter[str(mode)], accuracy_meter[str(mode)],
                    max(accuracy_meter[str(mode)].value(), best_perf)
                    if mode == '1' else None, green=True)

        # save checkpoint at the end of each train/val epoch
        #save_model(model, epoch + 1, accuracy_meter['validation'].value())



        #torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, root_path+"egtea/model.pth.tar")


def load_model(model):
    """
    load the saved state in the model passed as a parameter
    to load in main:
    model = BaselineModel(batch_size, seq_len, input_dim)
    model = model.to(device)
    load_model(model)

    :return:
    """
    chk = torch.load(root_path+"egtea/model.pth.tar")

    model.load_state_dict(chk['state_dict'])

def log(mode, epoch, loss_meter, accuracy_meter, best_perf=None, green=False):
    if mode == 0:
        mode = "Training"
    elif mode == 1:
        mode = "Validation"
    if green:
        print('\033[92m', end="")

    print(
        f"[{mode}] Epoch: {epoch:0.2f}. "
        f"Loss: {loss_meter.value():.2f}. "
        f"Accuracy: {accuracy_meter.value():.2f}% ", end="")

    if best_perf:
        print(f"[best: {best_perf:0.2f}]%", end="")

    print('\033[0m')



def label_smmothing(set_modality="standard", alpha=0.1, temperature = 0):
    """
    :param set_modality: standard or softmax or prior
    :param alpha:
    :param temperature: provided for softmax
    :return: smoothed labels depending on the specific modality
    """
    a = Glove(root_path+"Glove.6B/", alpha, set_modality, temperature)

    # print(a.find_similar("move")[1:6])
    b = a.get_ysoft()
    return b


if __name__ == '__main__':
    main()
