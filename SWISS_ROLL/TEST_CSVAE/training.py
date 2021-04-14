import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dists
import torch.utils.data as utils
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import sklearn.datasets 
from tqdm import tqdm, trange
import dill
import matplotlib.pyplot as plt

from model import *


def main():
    x_train, _ = sklearn.datasets.make_swiss_roll(1000)
    x_train = x_train.astype(np.float32)
    print(x_train.shape)
    y_train = (x_train[:, 0:1] >= 10).astype(np.float32)
    from sklearn.preprocessing import OneHotEncoder
    onehot_encoder = OneHotEncoder(sparse=False)
    abc = onehot_encoder.fit(y_train)

    z_dim = 2
    w_dim = 2
    batch_size = 32
    epochs = 100

    train_loader = torch.utils.data.DataLoader(list(zip(x_train, y_train)), shuffle=True, batch_size=batch_size)
    
    model = CSVAE(input_dim=x_train.shape[1], labels_dim=y_train.shape[1], z_dim=z_dim, w_dim=w_dim)
    
    params_without_delta = [param for name, param in model.named_parameters() if 'decoder_z_to_y' not in name]
    params_delta = [param for name, param in model.named_parameters() if 'decoder_z_to_y' in name]

    opt_without_delta = optim.Adam(params_without_delta, lr=(1e-3)/2)
    # scheduler_without_delta = optim.lr_scheduler.MultiStepLR(opt_without_delta, milestones=[pow(3, i) for i in range(7)], gamma=pow(0.1, 1/7))
    opt_delta = optim.Adam(params_delta, lr=(1e-3)/2)
    # scheduler_delta = optim.lr_scheduler.MultiStepLR(opt_delta, milestones=[pow(3, i) for i in range(7)], gamma=pow(0.1, 1/7))

    train_x_recon_losses = []
    # train_w_kl_losses = []
    # train_z_kl_losses = []
    # train_y_negentropy_losses = []
    # train_y_recon_losses = []

    for i in trange(epochs):
        for x, y in train_loader:

            from sklearn.preprocessing import OneHotEncoder
            onehot_encoder = OneHotEncoder(sparse=False)
            y_e = torch.FloatTensor(abc.transform(y))

            loss_val, x_recon_loss_val, w_kl_loss_val, z_kl_loss_val, y_negentropy_loss_val, y_recon_loss_val = model.calculate_loss(x, y, y_e)
            
            opt_delta.zero_grad()
            y_recon_loss_val.backward(retain_graph=True)

            opt_without_delta.zero_grad()
            loss_val.backward()
            
            
            opt_without_delta.step()
            opt_delta.step()
        
            train_x_recon_losses.append(x_recon_loss_val.item())
            # train_y_recon_losses.append(y_recon_loss_val.item())

    

    torch.save(model.state_dict(),  './results/model_csvae100.pt')

    colors_test = ['red' if label else 'blue' for label in y_train]

    x_mu, x_logvar, zw, _, _, _, _, _, _, _ = model.forward(torch.from_numpy(x_train), torch.from_numpy(y_train))

    # plt.plot(np.asarray(train_y_recon_losses))
    # plt.show() #plot the loss
    plt.plot(np.asarray(train_x_recon_losses))
    plt.show() #plot the loss
    # plt.plot(np.asarray(train_w_kl_losses))
    # plt.show() #plot the loss

    colors_train = ['green' if label else 'yellow' for label in y_train]
    x_mu = x_mu.detach().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_title('original')
    ax.scatter(x_train[:, 0], x_train[:, 1], x_train[:, 2], c=colors_train)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_title("reconstructed")
    ax.scatter(x_mu[:, 0], x_mu[:, 1], x_mu[:, 2], c=colors_train)
    plt.savefig('./results/recon_x_train')
    

    z_comp = zw[:, :2].detach().numpy()
    w_comp = zw[:, 2:].detach().numpy()
    
    scatter_size = 10

    cur_title = f'(z1, z2), epoch {i}'
    plt.figure(figsize=(5, 5,))
    plt.title(cur_title)
    plt.scatter(z_comp[:, 0], z_comp[:, 1], c=colors_test, s=scatter_size)
    plt.savefig('./plots100/1')

    cur_title = f'(z2, w1), epoch {i}'
    plt.figure(figsize=(5, 5,))
    plt.title(cur_title)
    plt.scatter(z_comp[:, 1], w_comp[:, 0], c=colors_test, s=scatter_size)
    plt.savefig('./plots100/2')

    cur_title = f'(w1, w2), epoch {i}'
    plt.figure(figsize=(5, 5,))
    plt.title(cur_title)
    plt.scatter(w_comp[:, 0], w_comp[:, 1], c=colors_test, s=scatter_size)
    plt.savefig('./plots100/3')

    cur_title = f'(w2, z1), epoch {i}'
    plt.figure(figsize=(5, 5,))
    plt.title(cur_title)
    plt.scatter(w_comp[:, 1], w_comp[:, 0], c=colors_test, s=scatter_size)
    plt.savefig('./plots100/4')

    plt.close('all')


if __name__ == "__main__":
    main()