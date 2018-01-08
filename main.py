from __future__ import print_function

import argparse
import os
import random
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torch.autograd import Variable

from data.read_data import Get_dataset
from models.Discriminator import _netD
from models.Generator import _netG

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="humanface",help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot' ,default="./face" , help='path to dataset')
parser.add_argument('--train_image_list',default='./data/filelist.txt',help='pics path lists')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=1000, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64 ,help='number of the filter of generate network')
parser.add_argument('--ndf', type=int, default=64,help='number of the filter of descriminator network')
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--decay_round', type=int, default=50, help='learning rate decay 0.5 each decay_round')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', type=bool,default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outpics', default='./pics', help='folder to output images')
parser.add_argument('--outckpts', default='./checkpoints', help='folder to output checkpoints')
parser.add_argument('--outlogs', default='./logs', help='folder to output images')
parser.add_argument('--manualSeed', type=int, help='manual seed')




def adjust_learning_rate(optimizers, epoch):
    """
    Sets the learning rate to the initial LR decayed by 10 every n epochs
    """
    lr = opt.lr * (0.5 ** (epoch // opt.decay_round))
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)





def main():

    ################ define global parameters #################
    global opt,label,real_label,fake_label,fixed_noise,noise,optimizerD,optimizerG,ndf,ngf,nz,nc,input

    #################  输出参数   ###############
    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    #####################################

    ############  构建结果保存的文件夹 #############
    try:
        os.makedirs(opt.outckpts)
        os.makedirs(opt.outpics)
        os.makedirs(opt.outlogs)
    except OSError:
        pass
    ###########################################
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    ##############   获取数据集   ############################
    print(opt.dataset)

    dataset = Get_dataset(dataset_name=opt.dataset, imageSize=opt.imageSize, data_dir=opt.dataroot,
                          image_list_file=opt.train_image_list).get_dataset()
    assert dataset

    ngpu = int(opt.ngpu)  # 使用多少GPU
    nz = int(opt.nz)  # 随机噪声的维数
    ngf = int(opt.ngf)  # G网络的第一层的filter数量
    ndf = int(opt.ndf)  # D网络的第一层的filter的数量
    nc = 3  # 图片的channel数量



    #######################  获得G网络的对象  ####################
    netG = _netG(ngpu=ngpu, nz=nz, ngf=ngf, nc=nc)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    ######################   获得D网络的对象  ######################
    netD = _netD(ngpu=ngpu,  ndf=ndf,nc=nc)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)



    input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
    #固定的噪声，用户生成测试用的图片，每个epoch训练完的网络使用相同的噪声来生成100张照片
    fixed_noise = torch.FloatTensor(100, nz, 1, 1).normal_(0, 1)
    label = torch.FloatTensor(opt.batchSize)
    real_label = 1
    fake_label = 0

    ################   定义loss函数     ########################
    mycriterion = nn.BCELoss()

    ##########################    将计算图放置到GPU     ######################
    if opt.cuda:
        netD.cuda()
        netG.cuda()
        mycriterion.cuda()
        input, label = input.cuda(), label.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    fixed_noise = Variable(fixed_noise)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    print("training is beginning .......................")
    traindataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,shuffle=True, num_workers=int(opt.workers))

    for epoch in range(opt.niter):

        adjust_learning_rate([optimizerG,optimizerD], epoch)

        train(traindataloader, epoch,netD=netD,netG=netG,criterion=mycriterion)



def train(train_loader, epoch,netD,netG,criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    Glosses = AverageMeter()
    Dlosses = AverageMeter()
    # switch to train mode
    netD.train()
    netG.train()

    start_time = time.time()
    log = ''
    for i, data in enumerate(train_loader, 0):
        data_time.update(time.time() - start_time)
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu,_= data
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        input.resize_as_(real_cpu).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        labelv = Variable(label)

        output = netD(inputv)
        errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        labelv = Variable(label.fill_(fake_label))

        output = netD(fake.detach())
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        Dlosses.update(errD)

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, labelv)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()

        ############################
        # (2) Update G network for the second time: maximize log(D(G(z)))
        ###########################
        fake = netG(noisev)
        netG.zero_grad()
        labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, labelv)
        errG.backward()
        D_G_z3 = output.data.mean()
        optimizerG.step()

        Glosses.update(errG)

        #更新一个batch的时间
        batch_time.update(time.time() - start_time)
        start_time = time.time()
        #输出信息
        log='[%d/%d][%d/%d]\t Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f /%.4f \tdatatime: %.4f \tbatchtime: %.4f' % (epoch, opt.niter, i, len(train_loader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2,D_G_z3 ,data_time.val,batch_time.val)
        print(log)
        #写日志
        logPath=opt.outlogs+'/%s_%d_%d_log.txt' % (opt.dataset,opt.batchSize,opt.imageSize)
        if not os.path.exists(logPath):
            fp = open(logPath, "w")

        with open(logPath, 'a+') as f:
            f.writelines(log + '\n')

######################################   存储记录等相关操作       #######################################3

    #5个epoch就生成一张图片
    if epoch % 5 == 0:
        # vutils.save_image(real_cpu,'%s/real_samples_epoch_%03d_batch%03d.png' % (opt.outpics, epoch,i), normalize=True)
        fake = netG(fixed_noise)
        vutils.save_image(fake.data,'%s/fake_epoch_%03d.png' % (opt.outpics, epoch),nrow=10,normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outckpts, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outckpts, epoch))


    print("one epoch time is===========================================",batch_time.sum)




class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



if __name__ == '__main__':
    main()