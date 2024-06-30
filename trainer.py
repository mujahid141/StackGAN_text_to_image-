import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import numpy as np
import torchfile
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from utils import *
from model import *
from dataset import *




class GANTrainer(object):
    def __init__(self, output_dir, args):
        if args.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            #self.summary_writer = FileWriter(self.log_dir)

        self.max_epoch = args.MAX_EPOCH
        self.snapshot_interval = args.SNAPSHOT_INTERVAL
        self.args = args

        s_gpus = args.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        self.batch_size = args.BATCH_SIZE * self.num_gpus
        self.output_dir = output_dir
        #print(self.gpus[0])
        #torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True
        
        # ############# For training stageI GAN #############
    def load_network_stageI(self):
        #from model import STAGE1_G, STAGE1_D
        netG = STAGE1_G(self.args)
        netG.apply(weights_init)
        print(netG)
        netD = STAGE1_D(self.args)
        netD.apply(weights_init)
        print(netD)
        print('***********************************************************')

        if self.args.NET_G != '':
            #state_dict = torch.load(args.NET_G, map_location=lambda storage, loc: storage)
            #netG.load_state_dict(state_dict)
            print('generator 1')
            print('Load from: ', self.args.NET_G)
        if self.args.NET_D != '':
            #state_dict = torch.load(args.NET_D,map_location=lambda storage, loc: storage)
            #netD.load_state_dict(state_dict)
            print('discriminator 1')
            print('Load from: ', args.NET_D)
        if self.args.CUDA:
            netG.cuda()
            netD.cuda()
        return netG, netD
        
    def load_network_stageII(self):
        #from model import STAGE1_G, STAGE2_G, STAGE2_D

        Stage1_G = STAGE1_G(self.args)
        netG = STAGE2_G(Stage1_G, self.args)
        netG.apply(weights_init)
        #print(netG)
        if self.args.NET_G != '':
            #state_dict = torch.load(args.NET_G,map_location=lambda storage, loc: storage)
            #netG.load_state_dict(state_dict)
            print('Load from: ', self.args.NET_G)
        elif self.args.STAGE1_G != '':
            #state_dict = torch.load(args.STAGE1_G,map_location=lambda storage, loc: storage)
            #netG.STAGE1_G.load_state_dict(state_dict)
            print('Load from: ', self.args.STAGE1_G)
        else:
            print("Please give the Stage1_G path")
            return

        netD = STAGE2_D(self.args)
        netD.apply(weights_init)
        if self.args.NET_D != '':
            #state_dict = torch.load(args.NET_D,map_location=lambda storage, loc: storage)
            #netD.load_state_dict(state_dict)
            print('Load from: ', args.NET_D)
        #print(netD)

        if self.args.CUDA:
            netG.cuda()
            netD.cuda()
        return netG, netD
    
    def train(self, data_loader, stage=1):
        if stage == 1:
            netG, netD = self.load_network_stageI()
        else:
            netG, netD = self.load_network_stageII()

        nz = self.args.Z_DIM
        batch_size = self.batch_size
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1),volatile=True)
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        if self.args.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
            real_labels, fake_labels = real_labels.cuda(), fake_labels.cuda()

        generator_lr = self.args.GENERATOR_LR
        discriminator_lr = self.args.DISCRIMINATOR_LR
        lr_decay_step = self.args.LR_DECAY_EPOCH
        optimizerD = optim.Adam(netD.parameters(), lr=self.args.DISCRIMINATOR_LR, betas=(0.5, 0.999))
        netG_para = []
        for p in netG.parameters():
            if p.requires_grad:
                netG_para.append(p)
        optimizerG = optim.Adam(netG_para,lr=self.args.GENERATOR_LR,betas=(0.5, 0.999))
        count = 0
        c = 0
        for epoch in range(self.max_epoch):
          start_t = time.time()
          if epoch % lr_decay_step == 0 and epoch > 0:
              generator_lr *= 0.5
              for param_group in optimizerG.param_groups:
                  param_group['lr'] = generator_lr
              discriminator_lr *= 0.5
              for param_group in optimizerD.param_groups:
                  param_group['lr'] = discriminator_lr
          br = 0
          for i, data in enumerate(data_loader, 0):
              if br == 3:
                  break
              ######################################################
              # (1) Prepare training data
              ######################################################
              print('data: -- ', data)
              print('---------------------')
              real_img_cpu, txt_embedding = data
              real_imgs = Variable(real_img_cpu)
              txt_embedding = Variable(txt_embedding)
              if self.args.CUDA:
                  real_imgs = real_imgs.cuda()
                  txt_embedding = txt_embedding.cuda()

              #######################################################
              # (2) Generate fake images
              ######################################################
              noise.data.normal_(0, 1)
              inputs = (txt_embedding, noise)
              _, fake_imgs, mu, logvar = \
                  nn.parallel.data_parallel(netG, inputs, self.gpus)

              ############################
              # (3) Update D network
              ###########################
              netD.zero_grad()
              errD, errD_real, errD_wrong, errD_fake = \
                  compute_discriminator_loss(netD, real_imgs, fake_imgs,
                                            real_labels, fake_labels,
                                            mu, self.gpus)
              errD.backward()
              optimizerD.step()
              ############################
              # (2) Update G network
              ###########################
              netG.zero_grad()
              errG = compute_generator_loss(netD, fake_imgs,
                                            real_labels, mu, self.gpus)
              kl_loss = KL_loss(mu, logvar)
              errG_total = errG + kl_loss * self.args.KL
              errG_total.backward()
              optimizerG.step()

              br = br + 1

              count = count + 1
              if i % 10 == 0:
                  inputs = (txt_embedding, fixed_noise)
                  lr_fake, fake, _, _ = nn.parallel.data_parallel(netG, inputs, self.gpus)
                  print('real_img_cpu', real_img_cpu.shape)
                  print(type(real_img_cpu))
                  print(real_img_cpu[0].shape)
                  plt.imshow(real_img_cpu[0].permute(1, 2, 0))
                  print('br', br)
          end_t = time.time()
          print('''[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_KL: %.4f
                  Loss_real: %.4f Loss_wrong:%.4f Loss_fake %.4f
                  Total Time: %.2fsec
                '''
                % (epoch, self.max_epoch, i, len(data_loader),
                  errD.data, errG.data, kl_loss.data,
                  errD_real, errD_wrong, errD_fake, (end_t - start_t)))

        #
        save_model(netG, netD, self.max_epoch, self.model_dir)
        #