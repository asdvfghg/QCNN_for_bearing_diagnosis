import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, recall_score, \
    precision_score, f1_score

import wandb
from torch import nn
from torch.utils.data import DataLoader

from Model.QCNN import QCNN
from Model.WDCNN import WDCNN
from utils.DatasetLoader import CustomTensorDataset
from utils.Preprocess import prepro
from utils.train_function import group_parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()





def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def select_model(chosen_model):
    if chosen_model == 'wdcnn':
        model = WDCNN()
    if chosen_model == 'qcnn':
        model = QCNN()
    return model


def train(config, dataloader):
    net = select_model(config.chosen_model)
    if use_gpu:
        net.cuda()
    wandb.watch(net, log="all")

    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []
    for e in range(config.epochs):
        for phase in ['train', 'validation']:
            loss = 0
            total = 0
            correct = 0
            loss_total = 0

            if phase == 'train':
                net.train()
            if phase == 'validation':
                net.eval()
                torch.no_grad()

            for step, (x, y) in enumerate(dataloader[phase]):

                x = x.type(torch.float)
                y = y.type(torch.long)
                y = y.view(-1)
                if use_gpu:
                    x, y = x.cuda(), y.cuda()
                if config.chosen_model == 'qcnn':
                    group = group_parameters(net)
                    optimizer = torch.optim.SGD([
                        {"params": group[0], "lr": config.lr},  # weight_r
                        {"params": group[1], "lr": config.lr * config.alpha},  # weight_g
                        {"params": group[2], "lr": config.lr * config.alpha},  # weight_b
                        {"params": group[3], "lr": config.lr},  # bias_r
                        {"params": group[4], "lr": config.lr * config.alpha},  # bias_g
                        {"params": group[5], "lr": config.lr * config.alpha},  # bias_b
                        {"params": group[6], "lr": config.lr},
                        {"params": group[7], "lr": config.lr},
                    ], lr=config.lr, momentum=0.9, weight_decay=1e-4)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 0.01)

                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs,
                                                                     eta_min=1e-8)  # goal: maximize Dice score


                else:
                    optimizer = torch.optim.SGD(net.parameters(), lr=config.lr, momentum=0.9)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs,
                                                                     eta_min=1e-8)
                loss_func = nn.CrossEntropyLoss()
                if use_gpu:
                    y_hat = net(x).cuda()
                else:
                    y_hat = net(x)
                loss = loss_func(y_hat, y)
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                loss_total += loss.item()
                y_predict = y_hat.argmax(dim=1)

                total += y.size(0)
                if use_gpu:
                    correct += (y_predict == y).cpu().squeeze().sum().numpy()
                else:
                    correct += (y_predict == y).squeeze().sum().numpy()

                if step % 20 == 0 and phase == 'train':
                    print('Epoch:%d, Step [%d/%d], Loss: %.4f'
                          % (
                          e + 1, step + 1, len(dataloader[phase].dataset), loss_total))
            # loss_total = loss_total / len(dataloader[phase].dataset)

            acc = correct / total
            if phase == 'train':
                train_loss.append(loss_total)
                train_acc.append(acc)
                wandb.log({
                    "Train Accuracy": 100. * acc,
                    "Train Loss": loss_total})
            if phase == 'validation':
                scheduler.step(loss_total)
                valid_loss.append(loss_total)
                valid_acc.append(acc)
                wandb.log({
                    "Validation Accuracy": 100. * acc,
                    "Validation Loss": loss_total})
            print('%s ACC:%.4f' % (phase, acc))
    return net


def inference(dataloader, model):
    net = model
    y_list, y_predict_list = [], []
    if use_gpu:
        net.cuda()
    net.eval()
    # endregion
    with torch.no_grad():
        for step, (x, y) in enumerate(dataloader):
            x = x.type(torch.float)
            y = y.type(torch.long)
            y = y.view(-1)
            if use_gpu:
                x, y = x.cuda(), y.cuda()
            y_hat = net(x)
            y_predict = y_hat.argmax(dim=1)
            y_list.extend(y.detach().cpu().numpy())
            y_predict_list.extend(y_predict.detach().cpu().numpy())

        cnf_matrix = confusion_matrix(y_list, y_predict_list)
        recall = recall_score(y_list, y_predict_list, average="macro")
        precision = precision_score(y_list, y_predict_list, average="macro")

        F1 = f1_score(y_list, y_predict_list, average="macro")
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)

        FP = FP.astype(float)
        TN = TN.astype(float)
        FPR = np.mean(FP / (FP + TN))
        wandb.log({
            "F1 Score": F1,
            "FPR": FPR,
            "Recall": recall,
            'PRE': precision})

        torch.save(net.state_dict(), os.path.join(wandb.run.dir, "checkpoint.pth"))
        wandb.save('*.pth')
        print('model saved')

        return F1


def main(config):
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    random_seed(config.seed)

    path = os.path.join('data', config.chosen_data)
    # train set, number denotes each category has 750 samples
    train_X, train_Y, valid_X, valid_Y = prepro(d_path=path,
                                                 length=2048,
                                                 number=750,
                                                 normal=False,
                                                 enc=True,
                                                 enc_step=28,
                                                 snr=config.snr,
                                                 property='Train',
                                                 noise=config.add_noise
                                                 )

    # test set, number denotes each category has 250 samples
    test_X, test_Y = prepro(d_path=path,
                             length=2048,
                             number=250,
                             normal=False,
                             enc=True,
                             enc_step=28,
                             snr=config.snr,
                             property='Test',
                             noise=config.add_noise
                             )

    train_X, valid_X, test_X = train_X[:, np.newaxis, :], valid_X[:, np.newaxis, :], test_X[:, np.newaxis, :]

    train_dataset = CustomTensorDataset(torch.tensor(train_X, dtype=torch.float), torch.tensor(train_Y))
    valid_dataset = CustomTensorDataset(torch.tensor(valid_X, dtype=torch.float), torch.tensor(valid_Y))
    test_dataset = CustomTensorDataset(torch.tensor(test_X, dtype=torch.float), torch.tensor(test_Y))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    data_loaders = {
        "train": train_loader,
        "validation": valid_loader
    }
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
    net = train(config, data_loaders)
    inference(test_loader, net)



if __name__ == '__main__':
    # wandb initialization, you need to create a wandb account and enter the username in 'entity'
    wandb.init(project="bearinganomaly", entity="")
    # WandB – Config is a variable that holds and saves hypermarkets and inputs
    config = wandb.config  # Initialize config
    config.no_cuda = False  # disables CUDA training
    config.log_interval = 200  # how many batches to wait before logging training status
    config.seed = 42  # random seed (default: 42)

    # Hyperparameters, lr and alpha need to fine-tune
    config.batch_size = 64  # input batch size for training (default: 64)
    config.epochs = 50  # number of epochs to train (default: 10)
    config.lr = 0.5  # learning rate (default: 0.5)
    config.alpha = 0.03 # scale factor alpha

    # noisy condition
    config.add_noise = True
    config.snr = -6

    # dataset and model
    config.chosen_data = '0HP'
    config.chosen_model = 'qcnn'  # wdcnn or qcnn

    main(config)
