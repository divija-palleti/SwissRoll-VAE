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
from sklearn.datasets import make_swiss_roll
from tqdm import tqdm, trange


class CSVAE(nn.Module):
    def __init__(self, input_dim, labels_dim, z_dim, w_dim):
        super(CSVAE, self).__init__()
        self.input_dim = input_dim
        self.labels_dim = labels_dim
        self.z_dim = z_dim
        self.w_dim = w_dim

        #encoder 
        self.encoder_xy_to_w = nn.Sequential(
            nn.Linear(input_dim+labels_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, 64), 
            nn.ReLU()
            )

        self.mu_xy_to_w = nn.Sequential(
            self.encoder_xy_to_w,
            nn.Linear(64, w_dim)
            )

        self.logvar_xy_to_w = nn.Sequential(
            self.encoder_xy_to_w,
            nn.Linear(64, w_dim)
            )
        
        self.encoder_x_to_z = nn.Sequential(
            nn.Linear(input_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, 64), 
            nn.ReLU()
            )

        self.mu_x_to_z = nn.Sequential(
            self.encoder_x_to_z,
            nn.Linear(64, z_dim)
            ) 

        self.logvar_x_to_z = nn.Sequential(
            self.encoder_x_to_z,
            nn.Linear(64, z_dim)
            ) 
        
        self.encoder_y_to_w = nn.Sequential(
            nn.Linear(labels_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, 64), 
            nn.ReLU()
            )

        self.mu_y_to_w = nn.Sequential(
            self.encoder_y_to_w,
            nn.Linear(64, w_dim)
            )
        
        self.logvar_y_to_w = nn.Sequential(
            self.encoder_y_to_w,
            nn.Linear(64, w_dim)
            )
        
        #decoder
        self.decoder_zw_to_x = nn.Sequential(
            nn.Linear(z_dim+w_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, 64), 
            nn.ReLU()
        )

        self.mu_zw_to_x = nn.Sequential(
            self.decoder_zw_to_x,
            nn.Linear(64, input_dim)
        )
        
        self.logvar_zw_to_x = nn.Sequential(
            self.decoder_zw_to_x,
            nn.Linear(64, input_dim)
        )

        self.decoder_z_to_y = nn.Sequential(
            nn.Linear(z_dim, 64), 
            nn.ReLU(), 
            nn.Linear(64, 64), 
            nn.ReLU(),
            nn.Linear(64, 2), 
            nn.Sigmoid()
        )

    def q_zw(self, x, y):

        xy = torch.cat([x, y], dim=1)
        
        z_mu = self.mu_x_to_z(x)
        z_logvar = self.logvar_x_to_z(x)

        w_mu_encoder = self.mu_xy_to_w(xy)
        w_logvar_encoder = self.logvar_xy_to_w(xy)
        
        w_mu_prior = self.mu_y_to_w(y)
        w_logvar_prior = self.logvar_y_to_w(y)
        
        return w_mu_encoder, w_logvar_encoder, w_mu_prior, w_logvar_prior, z_mu, z_logvar

    def p_x(self, z, w):    

        zw = torch.cat([z, w], dim=1)
        
        mu = self.mu_zw_to_x(zw)
        logvar = self.logvar_zw_to_x(zw)
        
        return mu, logvar

    def forward(self, x, y):
       
        w_mu_encoder, w_logvar_encoder, w_mu_prior, w_logvar_prior, z_mu, z_logvar = self.q_zw(x, y)
        w_encoder = self.reparameterize(w_mu_encoder, w_logvar_encoder)
        w_prior = self.reparameterize(w_mu_prior, w_logvar_prior)
        z = self.reparameterize(z_mu, z_logvar)
        zw = torch.cat([z, w_encoder], dim=1)
        
        x_mu, x_logvar = self.p_x(z, w_encoder)
        y_pred = self.decoder_z_to_y(z)
        
        return x_mu, x_logvar, zw, y_pred, w_mu_encoder, w_logvar_encoder, w_mu_prior, w_logvar_prior, z_mu, z_logvar

    def KL(mu1, mu2, logvar1, logvar2):
        std1 = torch.exp(0.5 * logvar1)
        std2 = torch.exp(0.5 * logvar2)
        return torch.sum(torch.log(std2) - torch.log(std1) + 0.5 * (torch.exp(logvar1) + (mu1 - mu2) ** 2) / torch.exp(logvar2) - 0.5, dim=-1)


    def calculate_loss(self, x, y, y_e):
  
        x_mu, x_logvar, zw, y_pred, w_mu_encoder, w_logvar_encoder, w_mu_prior, w_logvar_prior, z_mu, z_logvar = self.forward(x, y)
        
        x_recon = nn.MSELoss()(x_mu, x)
        
        w_dist = dists.MultivariateNormal(w_mu_encoder.flatten(), torch.diag(w_logvar_encoder.flatten().exp()))
        w_prior = dists.MultivariateNormal(w_mu_prior.flatten(), torch.diag(w_logvar_prior.flatten().exp()))
        w_kl = dists.kl.kl_divergence(w_dist, w_prior)
        
        z_dist = dists.MultivariateNormal(z_mu.flatten(), torch.diag(z_logvar.flatten().exp()))
        z_prior = dists.MultivariateNormal(torch.zeros(self.z_dim * z_mu.size()[0]), torch.eye(self.z_dim * z_mu.size()[0]))
        z_kl = dists.kl.kl_divergence(z_dist, z_prior)

        # print(type(w_logvar_encoder))
        # mu1, logvar1, mu2, logvar2

        # print(mu1.shape)
        # print(mu2.shape)
        # print(logvar1.shape)
        # print(logvar2.shape)
        # std1 = torch.exp(0.5 * logvar1)
        # 
        # return torch.sum(torch.log(std2) - torch.log(std1) + 0.5 * (torch.exp(logvar1) + (mu1 - mu2) ** 2) / torch.exp(logvar2) - 0.5, dim=-1)

        # mu1 = w_mu_encoder
        # mu2 = torch.zeros_like(w_mu_encoder)
        # logvar1 = w_logvar_encoder
        # logvar2 = torch.ones_like(logvar1) * 2.0
        # std2 = torch.exp(0.5 * logvar2)
        # std1 = torch.exp(0.5 * logvar1)
        # w_kl1 = torch.sum(torch.log(std2) - torch.log(std1) + 0.5 * (torch.exp(logvar1) + (mu1 - mu2) ** 2) / torch.exp(logvar2) - 0.5, dim=-1)

        # mu1 = w_mu_encoder
        # mu2 = torch.ones_like(w_mu_encoder)*3
        # logvar1 = w_logvar_encoder
        # logvar2 = torch.zeros_like(logvar1) 
        # std2 = torch.exp(0.5 * logvar2)
        # std1 = torch.exp(0.5 * logvar1)
        # w_kl0 = torch.sum(torch.log(std2) - torch.log(std1) + 0.5 * (torch.exp(logvar1) + (mu1 - mu2) ** 2) / torch.exp(logvar2) - 0.5, dim=-1)

        # y_pred_negentropy = torch.sum(y_pred.log() * y_pred )
        y_pred_negentropy = (y_pred.log() * y_pred + (1-y_pred).log() * (1-y_pred)).mean()
      
        # y_recon = nn.BCELoss()(y_pred, y_e) * -1
        y_recon = (100. * torch.where(y == 1, -torch.log(y_pred[:, 1]), -torch.log(y_pred[:, 0]))).mean()
        # alternatively use predicted logvar too to evaluate density of input
        
        # y_recon - optimized separately
        ELBO = 40 * x_recon + 0.2 * z_kl + 1 * w_kl + 110 * y_pred_negentropy
        # w_kl = torch.where(y == 1, w_kl1, w_kl0)

        # mu1 = z_mu
        # mu2 = torch.zeros_like(z_mu)
        # logvar1 = z_logvar
        # logvar2 = torch.zeros_like(z_logvar)
        # std2 = torch.exp(0.5 * logvar2)
        # std1 = torch.exp(0.5 * logvar1)
        # z_kl = torch.sum(torch.log(std2) - torch.log(std1) + 0.5 * (torch.exp(logvar1) + (mu1 - mu2) ** 2) / torch.exp(logvar2) - 0.5, dim=-1)




        # z_kl = self.KL(z_mu, z_logvar, torch.zeros_like(z_mu), torch.zeros_like(z_logvar))
        # ELBO = (20 * x_recon + 0.2 * z_kl + 1 * w_kl + 1000 * y_pred_negentropy).sum()
        
        return ELBO, x_recon, w_kl, z_kl, y_pred_negentropy, y_recon


    @staticmethod
    def reparameterize(mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(mu.device)
        return eps.mul(std).add_(mu)



