from __future__ import print_function

import argparse
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from data.read_data import Get_dataset
from models.HidingNet import HidingNet
from models.RevealNet import RevealNet
from models.HidingUNet import UnetGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="test", help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', default="/scratch/wxy/ImageNet/dataset/", help='path to dataset')
parser.add_argument('--train_image_list', default='./data/sub_image_list.txt', help='pics path lists')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
parser.add_argument('--decay_round', type=int, default=20, help='learning rate decay 0.5 each decay_round')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--Hnet', default='', help="path to Hidingnet (to continue training)")
parser.add_argument('--Rnet', default='', help="path to Revealnet (to continue training)")
parser.add_argument('--outpics', default='./training/pics', help='folder to output images')
parser.add_argument('--outckpts', default='./training/checkpoints', help='folder to output checkpoints')
parser.add_argument('--outlogs', default='./training/logs', help='folder to output images')
parser.add_argument('--beta', type=float, default=0.5, help='hyper parameter of β ')


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


# print the structure and parameters number of the net
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def main():
    ################ define global parameters #################
    global opt, optimizerH, optimizerR, writer

    writer = SummaryWriter()

    #################  输出参数   ###############
    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    #####################################

    ############  构建结果保存的文件夹 #############
    try:
        cur_time = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
        opt.outckpts += cur_time
        opt.outpics += cur_time
        opt.outlogs += cur_time
        if not os.path.exists(opt.outckpts):
            os.makedirs(opt.outckpts)
        if not os.path.exists(opt.outpics):
            os.makedirs(opt.outpics)
        if not os.path.exists(opt.outlogs):
            os.makedirs(opt.outlogs)
    except OSError:
        print("mkdir failed   XXXXXXXXXXXXXXXXXXXXX")
    ###########################################

    cudnn.benchmark = True

    ##############   获取数据集   ############################
    print(opt.dataset)

    dataset = Get_dataset(dataset_name=opt.dataset, imageSize=[opt.imageSize, opt.imageSize], data_dir=opt.dataroot,
                          image_list_file=opt.train_image_list).get_dataset()
    assert dataset

    ngpu = int(opt.ngpu)  # 使用多少GPU

    #######################  获得G网络的对象  ####################
    #
    Hnet=UnetGenerator(input_nc=6, output_nc=3, num_downs=7, ngf=64,norm_layer=nn.BatchNorm2d, use_dropout=False)
    Hnet.apply(weights_init)
    if opt.Hnet != '':
        Hnet.load_state_dict(torch.load(opt.Hnet))
    print_network(Hnet)

    ######################   获得D网络的对象  ######################
    Rnet = RevealNet(ngpu=ngpu)
    Rnet.apply(weights_init)
    if opt.Rnet != '':
        Rnet.load_state_dict(torch.load(opt.Rnet))
    print_network(Rnet)

    ################   定义L2 loss函数     ########################
    mycriterion = nn.MSELoss()

    ##########################    将计算图放置到GPU     ######################
    if opt.cuda:
        Hnet.cuda()
        Rnet.cuda()
        mycriterion.cuda()

    # setup optimizer
    optimizerH = optim.Adam(Hnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerR = optim.Adam(Rnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    print("training is beginning .......................")
    traindataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True,
                                                  num_workers=int(opt.workers))

    for epoch in range(opt.niter):
        adjust_learning_rate([optimizerH, optimizerR], epoch)

        train(traindataloader, epoch, Hnet=Hnet, Rnet=Rnet, criterion=mycriterion)

    writer.close()


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
    for i, data in enumerate(train_loader, 0):
        data_time.update(time.time() - start_time)

        Hnet.zero_grad()
        Rnet.zero_grad()
        allPics, _ = data
        this_batch_size = int(allPics.size(0) / 2)

        # 前面一半图片作为cover image ，后面一半图片作为secretImg
        coverImg = allPics[0:this_batch_size, :, :, :]
        secretImg = allPics[this_batch_size:this_batch_size * 2, :, :, :]

        # 将图片concat到一起，得到六通道图片作为H网络的输入
        concatImg = torch.cat([coverImg, secretImg], dim=1)

        # 数据放入GPU
        if opt.cuda:
            coverImg = coverImg.cuda()
            secretImg = secretImg.cuda()
            concatImg = concatImg.cuda()

        Hinputv = Variable(concatImg)
        originalLabelv = Variable(coverImg)

        ContainerImg = Hnet(Hinputv)  # 得到藏有secretimg的containerImg
        errH_original = criterion(ContainerImg, originalLabelv)  # Hiding net的重建误差
        Hlosses.update(errH_original.data[0], this_batch_size)  # 纪录H loss值

        # errH_original.backward()
        # optimizerH.step()  # 更新Hiding网络

        RevSecPic = Rnet(ContainerImg)
        secretLabelv = Variable(secretImg)  # label 为secret图片
        errR_secret = criterion(RevSecPic, secretLabelv)
        Rlosses.update(errR_secret.data[0], this_batch_size)  # 纪录R loss值

        # R网络的loss  乘以一个超参 β
        betaerrR_secret = opt.beta * errR_secret

        err_sum = errH_original + betaerrR_secret
        SumLosses.update(err_sum.data[0], this_batch_size)
        err_sum.backward()
        optimizerH.step()
        optimizerR.step()

        # 更新一个batch的时间
        batch_time.update(time.time() - start_time)
        start_time = time.time()

        # 输出信息
        log = '[%d/%d][%d/%d]\tLoss_H: %.4f Loss_R: %.4f Loss_sum: %.4f \tdatatime: %.4f \tbatchtime: %.4f' % (
            epoch, opt.niter, i, len(train_loader),
            Hlosses.val, Rlosses.val, SumLosses.val, data_time.val, batch_time.val)

        # 屏幕打印日志信息
        if i % 5 == 0:
            print(log)

        # 写日志
        logPath = opt.outlogs + '/%s_%d_%d_log.txt' % (opt.dataset, opt.batchSize, opt.imageSize)
        if not os.path.exists(logPath):
            fp = open(logPath, "w")

        with open(logPath, 'a+') as f:
            f.writelines(log + '\n')

        ######################################   存储记录等相关操作       #######################################3

        # 100个step就生成一张图片
        if epoch % 1 == 0 and i % 100 == 0:
            showContainer = torch.cat([originalLabelv.data, ContainerImg.data], 0)
            # vutils.save_image(showContainer, '%s/containers_epoch%03d_batch%04d.png' % (opt.outpics, epoch, i),
            #                   nrow=this_batch_size,
            #                   normalize=True)
            showReveal = torch.cat([secretLabelv.data, RevSecPic.data], 0)

            # vutils.save_image(showReveal, '%s/RevSecPics_epoch%03d_batch%04d.png' % (opt.outpics, epoch, i),
            #                   nrow=this_batch_size,
            #                   normalize=True)

            resultImg = torch.cat([showContainer, showReveal], 0)
            vutils.save_image(resultImg, '%s/ResultPics_epoch%03d_batch%04d.png' % (opt.outpics, epoch, i),
                              nrow=this_batch_size,
                              normalize=True)

    if epoch % 1 == 0:
        writer.add_scalar('train/R_loss', Rlosses.avg, epoch)
        writer.add_scalar('train/H_loss', Hlosses.avg, epoch)
        writer.add_scalar('train/sum_loss', SumLosses.avg, epoch)

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
