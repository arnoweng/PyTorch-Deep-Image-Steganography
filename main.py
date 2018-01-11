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
from PIL import Image
from torch.autograd import Variable

from data.read_data import Get_dataset
from models.Discriminator import Discriminator
from models.HidingNet import HidingNet
from models.RevealNet import RevealNet
from utils.transformed import to_tensor

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="test", help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', default="/scratch/wxy/ImageNet/dataset/", help='path to dataset')
parser.add_argument('--train_image_list', default='./data/image_list.txt', help='pics path lists')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--decay_round', type=int, default=50, help='learning rate decay 0.5 each decay_round')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--Hnet', default='', help="path to Hidingnet (to continue training)")
parser.add_argument('--Rnet', default='', help="path to Revealnet (to continue training)")
parser.add_argument('--outpics', default='./pics', help='folder to output images')
parser.add_argument('--outckpts', default='./checkpoints', help='folder to output checkpoints')
parser.add_argument('--outlogs', default='./logs', help='folder to output images')
parser.add_argument('--beta', type=float, default=1.0, help='hyper parameter of β ')


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
    global opt, label, real_label, fake_label, fixed_cover_with_sec, noise, optimizerH, optimizerR, ndf, ngf, nz, nc, input, secretLabel, originalLabel, secretImg

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

    cudnn.benchmark = True

    ##############   获取数据集   ############################
    print(opt.dataset)

    dataset = Get_dataset(dataset_name=opt.dataset, imageSize=[opt.imageSize, opt.imageSize], data_dir=opt.dataroot,
                          image_list_file=opt.train_image_list).get_dataset()
    assert dataset

    ngpu = int(opt.ngpu)  # 使用多少GPU

    #######################  获得G网络的对象  ####################
    Hnet = HidingNet(ngpu=ngpu)
    Hnet.apply(weights_init)
    if opt.Hnet != '':
        Hnet.load_state_dict(torch.load(opt.Hnet))
    print(Hnet)

    ######################   获得D网络的对象  ######################
    Rnet = RevealNet(ngpu=ngpu)
    Rnet.apply(weights_init)
    if opt.Rnet != '':
        Rnet.load_state_dict(torch.load(opt.Rnet))
    print(Rnet)

    input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)

    originalLabel = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)  # 原始的图片作为label
    secretLabel = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)  # 藏入的图片的label

    # # 固定的coverImg和secretImg cat到一起，用户生成测试用的图片，每个epoch训练完的网络使用相同的噪声来生成100张照片
    # fixed_cover_with_sec = torch.FloatTensor(1, 6, 128, 128)

    # real_label = 1
    # fake_label = 0

    secImgPath = "./secretImg/test.jpg"
    secretImg = Image.open(secImgPath).convert('RGB')
    secretImg = to_tensor(secretImg)

    ################   定义loss函数     ########################
    mycriterion = nn.MSELoss(size_average=False)

    ##########################    将计算图放置到GPU     ######################
    if opt.cuda:
        Hnet.cuda()
        Rnet.cuda()
        mycriterion.cuda()
        input, originalLabel = input.cuda(), originalLabel.cuda()
        secretLabel.cuda()

    # fixed_noise = Variable(fixed_noise)

    # setup optimizer
    optimizerH = optim.Adam(Hnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerR = optim.Adam(Rnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    print("training is beginning .......................")
    traindataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True,
                                                  num_workers=int(opt.workers))

    for epoch in range(opt.niter):
        adjust_learning_rate([optimizerH, optimizerR], epoch)

        train(traindataloader, epoch, Hnet=Hnet, Rnet=Rnet, criterion=mycriterion)


def train(train_loader, epoch, Hnet, Rnet, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    Hlosses = AverageMeter()
    Rlosses = AverageMeter()
    SumLosses = AverageMeter()

    # switch to train mode
    Hnet.train()
    Rnet.train()

    start_time = time.time()
    log = ''
    for i, data in enumerate(train_loader, 0):
        data_time.update(time.time() - start_time)

        Hnet.zero_grad()
        Rnet.zero_grad()
        concatPic, originalPic = data
        batch_size = concatPic.size(0)
        if opt.cuda:
            concatPic = concatPic.cuda()
            originalPic = originalPic.cuda()

        input.resize_as_(concatPic).copy_(concatPic)

        inputv = Variable(input)
        originalLabelv = Variable(originalPic)

        ContainerImg = Hnet(inputv)  # 得到藏有secretimg的containerImg
        errH_original = criterion(ContainerImg, originalLabelv)  # Hiding net的重建误差
        errH_original.backward(retain_graph=True)

        Hlosses.update(errH_original)  # 纪录H loss值

        RevSecPic = Rnet(ContainerImg)
        secretPic = torch.stack([secretImg])
        if opt.cuda:
            RevSecPic = RevSecPic.cuda()
            secretPic = secretPic.cuda()

        secretLabel = secretPic.expand(opt.batchSize, secretPic.size()[1], secretPic.size()[2], secretPic.size()[3])
        labelv = Variable(secretLabel)  # label 为secret图片
        errR_secret = criterion(RevSecPic, labelv)
        Rlosses.update(errR_secret)  # 纪录R loss值

        betaerrR_secret=opt.beta * errR_secret
        betaerrR_secret.backward()


        optimizerH.step()  # 更新Hiding网络
        optimizerR.step()  # 更新R

        err_sum = errH_original + betaerrR_secret

        SumLosses.update(err_sum)






        # 更新一个batch的时间
        batch_time.update(time.time() - start_time)
        start_time = time.time()
        # 输出信息
        log = '[%d/%d][%d/%d]\t Loss_H: %.4f Loss_R: %.4f Loss_sum: %.4f \tdatatime: %.4f \tbatchtime: %.4f' % (
            epoch, opt.niter, i, len(train_loader),
            Hlosses.val, Rlosses.val, SumLosses.val, data_time.val, batch_time.val)
        print(log)
        # 写日志
        logPath = opt.outlogs + '/%s_%d_%d_log.txt' % (opt.dataset, opt.batchSize, opt.imageSize)
        if not os.path.exists(logPath):
            fp = open(logPath, "w")

        with open(logPath, 'a+') as f:
            f.writelines(log + '\n')

        ######################################   存储记录等相关操作       #######################################3

        # 5个epoch就生成一张图片
        if epoch % 1 == 0 and i % 100 == 0:
            print(ContainerImg.data.size())
            print(originalLabelv.data.size())
            showContainer=torch.cat([originalLabelv.data, ContainerImg.data],0)
            print(showContainer.size())
            vutils.save_image(showContainer, '%s/containers_epoch_%03d_batch%03d.png' % (opt.outpics, epoch, i),
                              nrow=showContainer.size()[0]/2,
                              normalize=True)
            vutils.save_image(RevSecPic.data, '%s/RevSecPics_epoch_%03d_batch%03d.png' % (opt.outpics, epoch, i),
                              nrow=8,
                              normalize=True)

    if epoch % 5 == 0:
        # do checkpointing
        torch.save(Hnet.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outckpts, epoch))
        torch.save(Rnet.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outckpts, epoch))

    print("one epoch time is===========================================", batch_time.sum)


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
