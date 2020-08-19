import torch
from test import *
from Glove import Glove
import numpy as np
from Dataset import *
from Model import *
from torch.utils.data import DataLoader
from torch.nn import functional as F
from Utils import *
from smoothed_xent import SmoothedCrossEntropy
import pandas
from gaze_io_sample import *


root_path = "/home/2/2014/nagostin/Desktop/"


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("DEVICE= "+device)



# mode = "train"  # if train or test

# alpha = 0.2

#path_to_lmdb = [root_path + "egtea/TSN-C_3_egtea_action_CE_s1_rgb_model_best_fcfull_hd",root_path + "egtea/TSN-C_3_egtea_action_CE_s1_flow_model_best_fcfull_hd"]
#path_to_lmdb = [root_path + "egtea/TSN-C_3_egtea_action_CE_s1_rgb_model_best_fcfull_hd", root_path + "egtea/TSN-C_3_egtea_action_CE_s1_flow_model_best_fcfull_hd", root_path + "obj_FT"]  # the folders that contain the .mdb files

#path_to_lmdb = [root_path + "egtea/TSN-C_3_egtea_action_CE_s1_rgb_model_best_fcfull_hd"]
#path_to_lmdb = [root_path + "egtea/TSN-C_3_egtea_action_CE_s1_flow_model_best_fcfull_hd"]
#path_to_lmdb = [root_path + "obj_FT"]
#path_to_lmdb = [root_path + "egtea/TSN-C_3_egtea_action_CE_s1_rgb_model_best_fcfull_hd",root_path + "egtea/TSN-C_3_egtea_action_CE_s1_flow_model_best_fcfull_hd",root_path + "hand_obj_newfeat"]
#path_to_lmdb = [root_path + "egtea/TSN-C_3_egtea_action_CE_s1_rgb_model_best_fcfull_hd",root_path + "egtea/TSN-C_3_egtea_action_CE_s1_flow_model_best_fcfull_hd",root_path + "obj_54_FT"]
path_to_lmdb = [root_path + "hand_obj_newfeat"]
#path_to_lmdb = [root_path + "obj_54_FT"]
#path_to_lmdb = [root_path + "egtea/TSN-C_3_egtea_action_CE_s1_rgb_model_best_fcfull_hd",root_path + "obj_54_FT"]

### PATH OF TXT FOR TRAINING AND VALIDATION ###

groundtruth_path_train = [root_path+"egtea/action_annotation/train_split1.txt",
                          root_path+"egtea/action_annotation/train_split2.txt",
                          root_path+"egtea/action_annotation/train_split3.txt"]

groundtruth_path_test = [root_path+"egtea/action_annotation/test_split1.txt",
                          root_path+"egtea/action_annotation/test_split2.txt",
                          root_path+"egtea/action_annotation/test_split3.txt"]


###


path_to_csv_trainval = [root_path+"egtea/training1.csv", root_path+"egtea/validation1.csv"]  # path of train val csv

path_to_csv_test = root_path+"egtea/validation1.csv"  # for test dataloader


#experiment = "lr5_3br_ls"
saveModel = False
best = 0
mode = "train"

### SOME MODEL'S VARIABLES ###

input_dim = [1024, 1024, 54]
batch_size = 8
seq_len = 14

learning_rate = 0.00001


epochs = 60

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
    #correct_HM("/home/2/2014/nagostin/Desktop/Tesi/predictions/")  # postprocessing and extract new feature
    #split_train_val_detectron()
    #csv_to_txt()
    #split_frames_objDect("/Volumes/Bella_li/frames")

    #drawBBox()

    #test_gaze()
    #split_train_val_test_handMask(root_path+"hand14k/")
    #plot_gaze()
    #loadNPY()

    #show_8_Images()  # to print a example figure for thesis
    #plot_frequency_actions()  # to print a plot of frequency for thesis


    #generate_action_vnprior_csv()
    #generate_action_embeddings_csv()
    #upsample_to30fps("/home/2/2014/nagostin/Desktop/video/", "/home/2/2014/nagostin/Desktop/frames/")
    #upsample_to30fps("/Volumes/Bella_li/video/", "/Volumes/Bella_li/frames/")


    #inspect_lmdb("/volumes/Bella_li/egtea/TSN-C_3_egtea_action_CE_s1_flow_model_best_fcfull_hd/")

    #print_data(root_path + "hand_obj_newfeat")


    #split_frames_objDect("/Volumes/Bella_li/frames")
    #path = initialize_trainval_csv(1)  # to generate training and validation csv depending on split defined by authors of egtea gaze +

    #smoothed_labels = label_smoothing("glove")  # for smoothed labels




    model = BaselineModel(batch_size, seq_len, input_dim)

    model.to(device)
    print(model)

    #if mode == "train":
    data_loader_train = get_dataset(path_to_csv_trainval[0], batch_size, 4)  # loader for training
    data_loader_val = get_dataset(path_to_csv_trainval[1], batch_size, 4)  # loader for validation

    #data_loader_train = get_mock_dataloader()
    #data_loader_val = get_mock_dataloader()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    #criterion = SmoothedCrossEntropy(device=device, smooth_factor=0.2, smooth_prior="verb-noun", action_embeddings_csv_path="vn_prior.csv", reduce_time="mean")
    if mode == "train":
        train_val(model, [data_loader_train, data_loader_val], optimizer, epochs)  # with smoothed labels
    if mode == "test":
        data_loader_test = get_dataset(path_to_csv_test, batch_size, 4)
        epoch, perf, best_perf = load_model(model)  # load the best model saved
        print("Testing model" + str(best_perf))
        test_model(model, data_loader_test)

    #train_val(model, [data_loader_train, data_loader_val], optimizer, epochs)




def train_val(model, loaders, optimizer, epochs, resume = False):
    """

    :param model:
    :param loaders:
    :param optimizer:
    :param epochs:
    :return:
    """
    best = 0
    if resume == True:
        start_epoch, _, best_perf = load_model(model)
    else:
        best_perf = 0
    for epoch in range(epochs):
        #model.zero_grad()


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
                    #print(i)

                    x = batch['past_features']  # load in batch the next "past_features" datas of size (batch_size * 14 * 1024(352)

                    x = [xx.to(device) for xx in x]  # if input is a list (for multiple branch) then load in the device gpu

                    #print(x[0].size())

                    y = batch['label'].to(device)  # get the label of the batch (batch, 1)
                    #y = batch['label']


                    bs = y.shape[0]  # batch size

                    preds = model(x)



                    preds = preds.contiguous()

                    # linearize predictions
                    linear_preds = preds.view(-1, preds.shape[-1])  # (batch * 8 , 106)

                    #linear_labels = y.unsqueeze(1).expand(-1, preds.shape[1]).contiguous()  # for smoothed label

                    #print(linear_labels.size())
                    linear_labels = y.view(-1, 1).expand(-1, preds.shape[1]).contiguous().view(-1)

                    loss = F.cross_entropy(linear_preds, linear_labels)

                    #loss = criterion(preds, linear_labels)  # for smoothed labels

                    #print(loss)

                    # get the predictions for anticipation time = 1s (index -4) (anticipation)
                    # or for the last time-step (100%) (early recognition)
                    # top5 accuracy at 1s
                    idx = -4

                    k = 5  # top k = 5 anticipation

                    acc = topk_accuracy(preds[:, idx, :].detach().cpu().numpy(), y.detach().cpu().numpy(), (k,))[0] * 100  # top 5 accuracy percentage

                    #acc = topk_accuracy(preds[:, idx, :].detach().cpu().numpy(), y_temp.detach().cpu().numpy(), (k,))[0] * 100  # for smoothed labels

                    # store the values in the meters to keep incremental averages
                    loss_meter[str(mode)].add(loss.item(), bs)
                    accuracy_meter[str(mode)].add(acc, bs)


                    # if in training mode
                    if mode == 0:
                        optimizer.zero_grad()
                        loss.backward()
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
                    if mode == 1 else None, green=True)


                if accuracy_meter[str(mode)].value() > best_perf and mode == 1:  # if we are in validation and get an accuracy greater than before
                    best_perf = accuracy_meter[str(mode)].value()
                    if best_perf > best:
                        best = best_perf
                        save_model(model, epoch + 1, accuracy_meter['1'].value(), best_perf)


        # save checkpoint at the end of each train/val epoch
        #save_model(model, epoch + 1, accuracy_meter['validation'].value())

        if saveModel == True:
            save_model(model, epoch + 1, accuracy_meter[1].value(), best_perf)


def load_model(model, path_to_model = "/home/2/2014/nagostin/Desktop/egtea/model.pth.tar"):
    """
    load the saved state in the model passed as a parameter
    to load in main:
    model = BaselineModel(batch_size, seq_len, input_dim)
    model = model.to(device)
    load_model(model)

    :return:
    """
    chk = torch.load(path_to_model)

    #experiment = chk["experiment"]
    epoch = chk['epoch']
    best_perf = chk['best_perf']
    perf = chk['perf']

    model.load_state_dict(chk['state_dict'])

    return epoch, perf, best_perf

def save_model(model, epoch, perf, best_perf, path_to_model = "/home/2/2014/nagostin/Desktop/egtea/model.pth.tar"):

    torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'perf': perf, 'best_perf': best_perf}, path_to_model)



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



def label_smoothing(set_modality="glove", alpha=0.1, temperature = 0):
    """
    :param set_modality: standard or softmax or prior
    :param alpha:
    :param temperature: provided for softmax
    :return: smoothed labels depending on the specific modality
    """
    a = Glove(root_path+"Glove.6B/", alpha, set_modality, temperature)

    # print(a.find_similar("move")[1:6])
    b = a.get_ysoft()
    a.print_heatmap()
    return b

def generate_action_embeddings_csv():
    """
    generate the action_embeddings csv file loading glove from path, never mind about alpha, modality and temperature
    """
    path_of_glove = "/Users/nicolago/Desktop/Glove.6B/"
    a = Glove(path_of_glove)
    phi = a.get_phi()
    pandas.DataFrame(phi).to_csv("action_embeddings.csv")

def generate_action_vnprior_csv():
    """
    generate the action_embeddings csv file loading glove from path, never mind about alpha, modality and temperature
    """
    path_of_glove = "/Users/nicolago/Desktop/Glove.6B/"
    a = Glove(path_of_glove)
    prior = a.compute_vn_prior()
    pandas.DataFrame(prior).to_csv("vn_prior.csv")



def test_model(model, data_loader_test):
    """
    test the model in the current test set
    :return:
    """

    accuracy_meter = ValueMeter()

    with torch.set_grad_enabled(False):

        model.eval()

        for i, batch in enumerate(data_loader_test):
            # print(i)

            x = batch[
                'past_features']  # load in batch the next "past_features" datas of size (batch_size * 14 * 1024(352)

            x = [xx.to(device) for xx in x]  # if input is a list (for multiple branch) then load in the device gpu

            # print(x[0].size())

            y = batch['label'].to(device)  # get the label of the batch (batch, 1)
            # y = batch['label']

            bs = y.shape[0]  # batch size

            preds = model(x)

            preds = preds.contiguous()

            # get the predictions for anticipation time = 1s (index -4) (anticipation)
            # or for the last time-step (100%) (early recognition)
            # top5 accuracy at 1s
            idx = -4

            k = 5  # top k = 5 anticipation

            acc = topk_accuracy(preds[:, idx, :].detach().cpu().numpy(), y.detach().cpu().numpy(), (k,))[
                      0] * 100  # top 5 accuracy percentage

            # acc = topk_accuracy(preds[:, idx, :].detach().cpu().numpy(), y_temp.detach().cpu().numpy(), (k,))[0] * 100  # for smoothed labels

            # store the values in the meters to keep incremental averages
            accuracy_meter.add(acc, bs)

        # log at the end of testing
        print("top-k accuracy at 1 sec = " + accuracy_meter.value())


if __name__ == '__main__':
    main()
