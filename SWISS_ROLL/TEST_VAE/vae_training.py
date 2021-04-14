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
from sklearn.decomposition import PCA

from model import *

def main():
    pca_train = PCA(n_components=2)
    pca_test = PCA(n_components=2)
    x_train, _ = sklearn.datasets.make_swiss_roll(2000, random_state=207)
    x_train = x_train.astype(np.float32)
    pca_train.fit(x_train)
    x_train_pca = pca_train.transform(x_train)
    print(x_train.shape)
    y_train = (x_train[:, 0:1] >= 10).astype(np.float32)

    x_test, _ = sklearn.datasets.make_swiss_roll(500, random_state=101)
    x_test = x_test.astype(np.float32)
    print(x_test.shape)
    pca_test.fit(x_test)
    x_test_pca = pca_test.transform(x_test)
    y_test = (x_test[:, 0:1] >= 10).astype(np.float32)

    fig = plt.figure()

    colors_test = ['green' if label else 'yellow' for label in y_test]
    colors_train = ['green' if label else 'yellow' for label in y_train]
    scatter_size = 8
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('train')
    ax.scatter(x_train_pca[:, 0], x_train_pca[:, 1], c=colors_train, s=scatter_size)
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('test')
    ax.scatter(x_test_pca[:, 0], x_test_pca[:, 1], c=colors_test, s=scatter_size)
    plt.savefig('./results/intial_pca')


    # Parameters of the VAE
    d = 2  # latent space 
    D = input_dim = x_train.shape[1]
    activFunName = 'relu'  
    activations_list = {
        'softplus': nn.Softplus(),
        'tanh': nn.Tanh(),
        'relu': nn.ReLU()
    }
    activFun = activations_list[activFunName]
    H1 = 100
    H2 = 100
    lambda_reg = 1e-3  # For the weights of the networks
    epoch = 100
    initial = int(0.33 * epoch)
    learning_rate = 1e-3
    clipping_value = 1
    batch_size = 100

    train_loader = torch.utils.data.DataLoader(list(zip(x_train, y_train)), shuffle=True, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(list(zip(x_test, y_test)), shuffle=True, batch_size=batch_size)
    model = VAE_model(d, D, H1, H2, activFun)
    optimizer_model = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lambda_reg)
    
    trainingLoss = []

    ELBO = np.zeros((epoch, 1))
    for i in range(epoch):
        # Initialize the losses
        train_loss = 0
        train_loss_num = 0
        for batch_idx, (x, y) in enumerate(train_loader):

            MU_X_eval, LOG_VAR_X_eval, Z_ENC_eval, MU_Z_eval, LOG_VAR_Z_eval = model(x)

             # Compute the regularization parameter
            if initial == 0:
                r = 0
            else:
                r = 1. * i / initial
                if r > 1.:
                    r = 1.
            
             # The VAE loss

            loss = model.VAE_loss(x=x, mu_x=MU_X_eval, log_var_x=LOG_VAR_X_eval, mu_z=MU_Z_eval, log_var_z=LOG_VAR_Z_eval, r=r)

            # Update the parameters
            optimizer_model.zero_grad()

            # Compute the loss
            loss.backward()

            # Update the parameters
            optimizer_model.step()

             # Collect the ways
            train_loss += loss.item()
            train_loss_num += 1

        ELBO[i] = train_loss / train_loss_num
        if i % 10 == 0:
            print("[Epoch: {}/{}] [objective: {:.3f}]".format(i, epoch, ELBO[i, 0]))


    ELBO_train = ELBO[epoch-1, 0].round(2)
    print('[ELBO train: ' + str(ELBO_train) + ']')
    del MU_X_eval, MU_Z_eval, Z_ENC_eval
    del LOG_VAR_X_eval, LOG_VAR_Z_eval
    print("Training finished")

    plt.figure()
    plt.plot(ELBO)
    plt.show()      

    torch.save(model.state_dict(),  './results/model_swiss_roll_VAE.pt')

    MU_X_eval, LOG_VAR_X_eval, Z_ENC_eval, MU_Z_eval, LOG_VAR_Z_eval = model(torch.from_numpy(x_train))
    MU_X_eval = MU_X_eval.detach().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.set_title('original')
    ax.scatter(x_train[:, 0], x_train[:, 1], x_train[:, 2], c=colors_train, s=scatter_size)
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.set_title('recon')
    ax.scatter(MU_X_eval[:, 0], MU_X_eval[:, 1], MU_X_eval[:, 2], c=colors_train, s=scatter_size)
    ax = fig.add_subplot(2, 2, 3)
    ax.set_title('original')
    ax.scatter(x_train_pca[:, 0], x_train_pca[:, 1], c=colors_train, s=scatter_size)
    x_train_pca_recon = pca_train.transform(MU_X_eval)
    ax = fig.add_subplot(2, 2, 4)
    ax.set_title('recon')
    ax.scatter(x_train_pca_recon[:, 0], x_train_pca_recon[:, 1], c=colors_train, s=scatter_size)
    plt.savefig('./results/recon_x_train')

    MU_X_eval, LOG_VAR_X_eval, Z_ENC_eval, MU_Z_eval, LOG_VAR_Z_eval = model(torch.from_numpy(x_test))
    MU_X_eval = MU_X_eval.detach().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(3, 2, 1, projection='3d')
    ax.set_title('original')
    ax.scatter(x_test[:, 0], x_test[:, 1], x_test[:, 2], c=colors_test, s=scatter_size)
    ax = fig.add_subplot(3, 2, 2, projection='3d')
    ax.set_title('recon')
    ax.scatter(MU_X_eval[:, 0], MU_X_eval[:, 1], MU_X_eval[:, 2], c=colors_test, s=scatter_size)
    ax = fig.add_subplot(2, 2, 3)
    ax.set_title('original')
    ax.scatter(x_test_pca[:, 0], x_test_pca[:, 1], c=colors_test, s=scatter_size)
    x_test_pca_recon = pca_test.transform(MU_X_eval)
    ax = fig.add_subplot(2, 2, 4)
    ax.set_title('recon')
    ax.scatter(x_test_pca_recon[:, 0], x_test_pca_recon[:, 1], c=colors_test, s=scatter_size)
    plt.savefig('./results/recon_x_test')

#     colors_test = ['red' if label else 'blue' for label in y_train]

#     _, _, zw, _, _, _, _, _, _, _ = model.forward(torch.from_numpy(x_train), torch.from_numpy(y_train))

#     plt.plot(np.asarray(train_y_recon_losses))
#     plt.show() #plot the loss
#     # plt.plot(np.asarray(train_x_recon_losses))
#     # plt.show() #plot the loss
#     # plt.plot(np.asarray(train_w_kl_losses))
#     # plt.show() #plot the loss
    

#     z_comp = zw[:, :2].detach().numpy()
#     w_comp = zw[:, 2:].detach().numpy()
    
#     scatter_size = 12

#     cur_title = f'(z1, z2), epoch {i}'
#     plt.figure(figsize=(5, 5,))
#     plt.title(cur_title)
#     plt.scatter(z_comp[:, 0], z_comp[:, 1], c=colors_test, s=scatter_size)
#     plt.savefig('./plots1000/1')

#     cur_title = f'(z2, w1), epoch {i}'
#     plt.figure(figsize=(5, 5,))
#     plt.title(cur_title)
#     plt.scatter(z_comp[:, 1], w_comp[:, 0], c=colors_test, s=scatter_size)
#     plt.savefig('./plots1000/2')

#     cur_title = f'(w1, w2), epoch {i}'
#     plt.figure(figsize=(5, 5,))
#     plt.title(cur_title)
#     plt.scatter(w_comp[:, 0], w_comp[:, 1], c=colors_test, s=scatter_size)
#     plt.savefig('./plots1000/3')

#     cur_title = f'(w2, z1), epoch {i}'
#     plt.figure(figsize=(5, 5,))
#     plt.title(cur_title)
#     plt.scatter(w_comp[:, 1], w_comp[:, 0], c=colors_test, s=scatter_size)
#     plt.savefig('./plots1000/4')

#     plt.close('all')


if __name__ == "__main__":
    main()