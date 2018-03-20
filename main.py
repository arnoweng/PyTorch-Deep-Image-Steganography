# encoding: utf-8
"""
@author: yongzhi li
@contact: yongzhili@vip.qq.com

@version: 1.0
@file: main.py
@time: 2018/3/20

"""

import argparse
import os
import shutil
import socket
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import utils.transformed as transforms
from data.ImageFolderDataset import MyImageFolder
from models.HidingUNet import UnetGenerator
from models.RevealNet import RevealNet

DATA_DIR = '/n/liyz/data/deep-steganography-dataset/'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="train",
                    help='train | val | test')
parser.add_argument('--workers', type=int, default=8,
                    help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=32,
                    help='input batch size')
parser.add_argument('--imageSize', type=int, default=256,
                    help='the number of frames')
parser.add_argument('--niter', type=int, default=100,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate, default=0.001')
parser.add_argument('--decay_round', type=int, default=10,
                    help='learning rate decay 0.5 each decay_round')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', type=bool, default=True,
                    help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--Hnet', default='',
                    help="path to Hidingnet (to continue training)")
parser.add_argument('--Rnet', default='',
                    help="path to Revealnet (to continue training)")
parser.add_argument('--trainpics', default='./training/',
                    help='folder to output training images')
parser.add_argument('--validationpics', default='./training/',
                    help='folder to output validation images')
parser.add_argument('--testPics', default='./training/',
                    help='folder to output test images')
parser.add_argument('--outckpts', default='./training/',
                    help='folder to output checkpoints')
parser.add_argument('--outlogs', default='./training/',
                    help='folder to output images')
parser.add_argument('--outcodes', default='./training/',
                    help='folder to save the experiment codes')
parser.add_argument('--beta', type=float, default=0.75,
                    help='hyper parameter of beta')
parser.add_argument('--remark', default='', help='comment')
parser.add_argument('--test', default='', help='test mode, you need give the test pics dirs in this param')

parser.add_argument('--hostname', default=socket.gethostname(), help='the  host name of the running server')
parser.add_argument('--debug', type=bool, default=False, help='debug mode do not create folders')
parser.add_argument('--logFrequency', type=int, default=10, help='the frequency of print the log on the console')
parser.add_argument('--resultPicFrequency', type=int, default=100, help='the frequency of save the resultPic')


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
    print_log(str(net), logPath)
    print_log('Total number of parameters: %d' % num_params, logPath)


# 保存本次实验的代码
def save_current_codes(des_path):
    main_file_path = os.path.realpath(__file__)  # eg：/n/liyz/videosteganography/main.py
    cur_work_dir, mainfile = os.path.split(main_file_path)  # eg：/n/liyz/videosteganography/

    new_main_path = os.path.join(des_path, mainfile)
    shutil.copyfile(main_file_path, new_main_path)

    data_dir = cur_work_dir + "/data/"
    new_data_dir_path = des_path + "/data/"
    shutil.copytree(data_dir, new_data_dir_path)

    model_dir = cur_work_dir + "/models/"
    new_model_dir_path = des_path + "/models/"
    shutil.copytree(model_dir, new_model_dir_path)

    utils_dir = cur_work_dir + "/utils/"
    new_utils_dir_path = des_path + "/utils/"
    shutil.copytree(utils_dir, new_utils_dir_path)


def main():
    ############### define global parameters ###############
    global opt, optimizerH, optimizerR, writer, logPath, schedulerH, schedulerR, val_loader, smallestLoss

    #################  输出配置参数   ###############
    opt = parser.parse_args()

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")

    cudnn.benchmark = True

    ############  create the dirs to save the result #############
    if not opt.debug:
        try:
            cur_time = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
            experiment_dir = opt.hostname + "_" + cur_time + opt.remark
            opt.outckpts += experiment_dir + "/checkPoints"
            opt.trainpics += experiment_dir + "/trainPics"
            opt.validationpics += experiment_dir + "/validationPics"
            opt.outlogs += experiment_dir + "/trainingLogs"
            opt.outcodes += experiment_dir + "/codes"
            opt.testPics += experiment_dir + "/testPics"
            if not os.path.exists(opt.outckpts):
                os.makedirs(opt.outckpts)
            if not os.path.exists(opt.trainpics):
                os.makedirs(opt.trainpics)
            if not os.path.exists(opt.validationpics):
                os.makedirs(opt.validationpics)
            if not os.path.exists(opt.outlogs):
                os.makedirs(opt.outlogs)
            if not os.path.exists(opt.outcodes):
                os.makedirs(opt.outcodes)
            if (not os.path.exists(opt.testPics)) and opt.test != '':
                os.makedirs(opt.testPics)

        except OSError:
            print("mkdir failed   XXXXXXXXXXXXXXXXXXXXX")

    logPath = opt.outlogs + '/%s_%d_log.txt' % (opt.dataset, opt.batchSize)

    # 保存模型的参数
    print_log(str(opt), logPath)
    # 保存本次实验的代码
    save_current_codes(opt.outcodes)

    if opt.test == '':
        # tensorboardX writer
        writer = SummaryWriter(comment='**' + opt.remark)
        ##############   获取数据集   ############################
        traindir = os.path.join(DATA_DIR, 'train')
        valdir = os.path.join(DATA_DIR, 'val')
        train_dataset = MyImageFolder(
            traindir,  # 对数据进行预处理
            transforms.Compose([  # 将几个transforms 组合在一起
                transforms.Resize([opt.imageSize, opt.imageSize]),  # 随机切再resize成给定的size大小
                transforms.ToTensor(),
                # 把一个取值范围是[0,255]或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
            ]))
        val_dataset = MyImageFolder(
            valdir,  # 对数据进行预处理
            transforms.Compose([  # 将几个transforms 组合在一起
                transforms.Resize([opt.imageSize, opt.imageSize]),  # 随机切再resize成给定的size大小
                transforms.ToTensor(),  # 把一个取值范围是[0,255]或者shape为(H,W,C)的numpy.ndarray，
                # 转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
            ]))
        assert train_dataset
        assert val_dataset
    else:
        opt.Hnet = "./checkPoint/netH_epoch_73,sumloss=0.000447,Hloss=0.000258.pth"
        opt.Rnet = "./checkPoint/netR_epoch_73,sumloss=0.000447,Rloss=0.000252.pth"
        testdir = opt.test
        test_dataset = MyImageFolder(
            testdir,  # 对数据进行预处理
            transforms.Compose([  # 将几个transforms 组合在一起
                transforms.Resize([opt.imageSize, opt.imageSize]),  # 随机切再resize成给定的size大小
                transforms.ToTensor(),
                # 把一个取值范围是[0,255]或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
            ]))
        assert test_dataset

    ##################  获得Hiding网络的对象  #################
    # Hnet = UnetGenerator(input_nc=144, output_nc=72, num_downs=7, ngf=128,
    #                      norm_layer=nn.BatchNorm2d, use_dropout=False)

    Hnet = UnetGenerator(input_nc=6, output_nc=3, num_downs=7, output_function=nn.Sigmoid)
    Hnet.cuda()
    Hnet.apply(weights_init)
    # 判断是否接着之前的训练
    if opt.Hnet != "":
        Hnet.load_state_dict(torch.load(opt.Hnet))
    # 两块卡加这行
    if opt.ngpu > 1:
        Hnet = torch.nn.DataParallel(Hnet).cuda()
    print_network(Hnet)

    ################   获得Reveal网络的对象  ################
    Rnet = RevealNet(output_function=nn.Sigmoid)
    Rnet.cuda()
    Rnet.apply(weights_init)
    if opt.Rnet != '':
        Rnet.load_state_dict(torch.load(opt.Rnet))
    if opt.ngpu > 1:
        Rnet = torch.nn.DataParallel(Rnet).cuda()
    print_network(Rnet)

    # MSE loss
    criterion = nn.MSELoss().cuda()
    # training mode
    if opt.test == '':
        # setup optimizer
        optimizerH = optim.Adam(Hnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        schedulerH = ReduceLROnPlateau(optimizerH, mode='min', factor=0.2, patience=5, verbose=True)

        optimizerR = optim.Adam(Rnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        schedulerR = ReduceLROnPlateau(optimizerR, mode='min', factor=0.2, patience=8, verbose=True)

        train_loader = DataLoader(train_dataset, batch_size=opt.batchSize,
                                  shuffle=True, num_workers=int(opt.workers))
        val_loader = DataLoader(val_dataset, batch_size=opt.batchSize,
                                shuffle=False, num_workers=int(opt.workers))
        smallestLoss = 10000
        print_log("training is beginning .......................................................", logPath)
        for epoch in range(opt.niter):
            ######################## train ##########################################
            train(train_loader, epoch, Hnet=Hnet, Rnet=Rnet, criterion=criterion)

            ####################### validation  #####################################
            val_hloss, val_rloss, val_sumloss = validation(val_loader, epoch, Hnet=Hnet, Rnet=Rnet, criterion=criterion)

            ####################### adjust learning rate ############################
            schedulerH.step(val_sumloss)
            schedulerR.step(val_rloss)

            # save the best model parameters
            if val_sumloss < globals()["smallestLoss"]:
                globals()["smallestLoss"] = val_sumloss
                # do checkPointing
                torch.save(Hnet.state_dict(),
                           '%s/netH_epoch_%d,sumloss=%.6f,Hloss=%.6f.pth' % (
                               opt.outckpts, epoch, val_sumloss, val_hloss))
                torch.save(Rnet.state_dict(),
                           '%s/netR_epoch_%d,sumloss=%.6f,Rloss=%.6f.pth' % (
                               opt.outckpts, epoch, val_sumloss, val_rloss))

        writer.close()

     # test mode
    else:
        test_loader = DataLoader(test_dataset, batch_size=opt.batchSize,
                                 shuffle=False, num_workers=int(opt.workers))
        test(test_loader, 0, Hnet=Hnet, Rnet=Rnet, criterion=criterion)
        print("##################   test is completed, the result pic is saved in the ./training/yourcompuer+time/testPics/   ######################")


def train(train_loader, epoch, Hnet, Rnet, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    Hlosses = AverageMeter()  # 纪录每个epoch H网络的loss
    Rlosses = AverageMeter()  # 纪录每个epoch R网络的loss
    SumLosses = AverageMeter()  # 纪录每个epoch Hloss + β*Rloss

    # switch to train mode
    Hnet.train()
    Rnet.train()

    start_time = time.time()
    for i, data in enumerate(train_loader, 0):
        data_time.update(time.time() - start_time)

        Hnet.zero_grad()
        Rnet.zero_grad()

        all_pics = data  # allpics包含coverImg 和 secretImg,不需要label
        this_batch_size = int(all_pics.size()[0] / 2)  # 处理每个epoch 最后一个batch可能不足opt.bachsize

        # 前面一半图片作为coverImg ，后面一半图片作为secretImg
        cover_img = all_pics[0:this_batch_size, :, :, :]  # batchsize,3,256,256
        secret_img = all_pics[this_batch_size:this_batch_size * 2, :, :, :]

        # 将图片concat到一起，得到六通道图片作为H网络的输入
        concat_img = torch.cat([cover_img, secret_img], dim=1)

        # 数据放入GPU
        if opt.cuda:
            cover_img = cover_img.cuda()
            secret_img = secret_img.cuda()
            concat_img = concat_img.cuda()

        concat_imgv = Variable(concat_img)  # concatImg 作为H网络的输入
        cover_imgv = Variable(cover_img)  # coverImg 作为H网络的label

        container_img = Hnet(concat_imgv)  # 得到藏有secretimg的containerImg
        errH = criterion(container_img, cover_imgv)  # Hiding net的重建误差
        Hlosses.update(errH.data[0], this_batch_size)  # 纪录H loss值

        rev_secret_img = Rnet(container_img)  # containerImg作为R网络的输入 得到RevSecImg
        secret_imgv = Variable(secret_img)  # secretImg作为R网络的label
        errR = criterion(rev_secret_img, secret_imgv)  # Reveal net的重建误差
        Rlosses.update(errR.data[0], this_batch_size)  # 纪录R loss值

        betaerrR_secret = opt.beta * errR
        err_sum = errH + betaerrR_secret
        SumLosses.update(err_sum.data[0], this_batch_size)
        # 计算梯度
        err_sum.backward()

        # 优化两个网络的参数
        optimizerH.step()
        optimizerR.step()

        # 更新一个batch的时间
        batch_time.update(time.time() - start_time)
        start_time = time.time()

        # 日志信息
        log = '[%d/%d][%d/%d]\tLoss_H: %.4f Loss_R: %.4f Loss_sum: %.4f \tdatatime: %.4f \tbatchtime: %.4f' % (
            epoch, opt.niter, i, len(train_loader),
            Hlosses.val, Rlosses.val, SumLosses.val, data_time.val, batch_time.val)

        # 屏幕打印日志信息
        if i % opt.logFrequency == 0:
            print_log(log, logPath)
        else:
            print_log(log, logPath, console=False)

        #######################################   存储记录等相关操作       #######################################3
        # 100个step就生成一张图片
        if epoch % 1 == 0 and i % opt.resultPicFrequency == 0:
            save_result_pic(this_batch_size, cover_img, container_img.data, secret_img, rev_secret_img.data, epoch, i,
                            opt.trainpics)

    # 输出一个epoch所用时间
    epoch_log = "one epoch time is %.4f======================================================================" % (
        batch_time.sum) + "\n"
    epoch_log = epoch_log + "epoch learning rate: optimizerH_lr = %.8f      optimizerR_lr = %.8f" % (
        optimizerH.param_groups[0]['lr'], optimizerR.param_groups[0]['lr']) + "\n"
    epoch_log = epoch_log + "epoch_Hloss=%.6f\tepoch_Rloss=%.6f\tepoch_sumLoss=%.6f" % (
        Hlosses.avg, Rlosses.avg, SumLosses.avg)
    print_log(epoch_log, logPath)

    if not opt.debug:
        # 纪录learning rate
        writer.add_scalar("lr/H_lr", optimizerH.param_groups[0]['lr'], epoch)
        writer.add_scalar("lr/R_lr", optimizerR.param_groups[0]['lr'], epoch)
        writer.add_scalar("lr/beta", opt.beta, epoch)
        # 每个epoch纪录一次平均loss 在tensorboard展示
        writer.add_scalar('train/R_loss', Rlosses.avg, epoch)
        writer.add_scalar('train/H_loss', Hlosses.avg, epoch)
        writer.add_scalar('train/sum_loss', SumLosses.avg, epoch)


def validation(val_loader, epoch, Hnet, Rnet, criterion):
    print(
        "#################################################### validation begin ########################################################")
    start_time = time.time()
    Hnet.eval()
    Rnet.eval()
    Hlosses = AverageMeter()  # 纪录每个epoch H网络的loss
    Rlosses = AverageMeter()  # 纪录每个epoch R网络的loss
    for i, data in enumerate(val_loader, 0):
        Hnet.zero_grad()
        Rnet.zero_grad()
        all_pics = data  # allpics包含coverImg 和 secretImg,不需要label
        this_batch_size = int(all_pics.size()[0] / 2)  # 处理每个epoch 最后一个batch可能不足opt.bachsize

        # 前面一半图片作为coverImg ，后面一半图片作为secretImg
        cover_img = all_pics[0:this_batch_size, :, :, :]  # batchsize,3,256,256
        secret_img = all_pics[this_batch_size:this_batch_size * 2, :, :, :]

        # 将图片concat到一起，得到六通道图片作为H网络的输入
        concat_img = torch.cat([cover_img, secret_img], dim=1)

        # 数据放入GPU
        if opt.cuda:
            cover_img = cover_img.cuda()
            secret_img = secret_img.cuda()
            concat_img = concat_img.cuda()

        concat_imgv = Variable(concat_img, volatile=True)  # concatImg 作为H网络的输入
        cover_imgv = Variable(cover_img, volatile=True)  # coverImg 作为H网络的label

        container_img = Hnet(concat_imgv)  # 得到藏有secretimg的containerImg
        errH = criterion(container_img, cover_imgv)  # Hiding net的重建误差
        Hlosses.update(errH.data[0], this_batch_size)  # 纪录H loss值

        rev_secret_img = Rnet(container_img)  # containerImg作为R网络的输入 得到RevSecImg
        secret_imgv = Variable(secret_img, volatile=True)  # secretImg作为R网络的label
        errR = criterion(rev_secret_img, secret_imgv)  # Reveal net的重建误差
        Rlosses.update(errR.data[0], this_batch_size)  # 纪录R loss值

        if i % 50 == 0:
            save_result_pic(this_batch_size, cover_img, container_img.data, secret_img, rev_secret_img.data, epoch, i,
                            opt.validationpics)

    val_hloss = Hlosses.avg
    val_rloss = Rlosses.avg
    val_sumloss = val_hloss + opt.beta * val_rloss

    val_time = time.time() - start_time
    val_log = "validation[%d] val_Hloss = %.6f\t val_Rloss = %.6f\t val_Sumloss = %.6f\t validation time=%.2f" % (
        epoch, val_hloss, val_rloss, val_sumloss, val_time)
    print_log(val_log, logPath)

    if not opt.debug:
        writer.add_scalar('validation/H_loss_avg', Hlosses.avg, epoch)
        writer.add_scalar('validation/R_loss_avg', Rlosses.avg, epoch)
        writer.add_scalar('validation/sum_loss_avg', val_sumloss, epoch)

    print(
        "#################################################### validation end ########################################################")
    return val_hloss, val_rloss, val_sumloss


def test(test_loader, epoch, Hnet, Rnet, criterion):
    print(
        "#################################################### test begin ########################################################")
    start_time = time.time()
    Hnet.eval()
    Rnet.eval()
    Hlosses = AverageMeter()  # to record the Hloss in one epoch
    Rlosses = AverageMeter()  # to record the Rloss in one epoch
    for i, data in enumerate(test_loader, 0):
        Hnet.zero_grad()
        Rnet.zero_grad()
        all_pics = data  # allpics contian cover_img and secret_img ,label is not needed
        this_batch_size = int(all_pics.size()[0] / 2)  # in order to handle the final batch which may not have opt.size

        # half of the front is as cover_img ，half of the end is as secret_img
        cover_img = all_pics[0:this_batch_size, :, :, :]  # batchSize,3,256,256
        secret_img = all_pics[this_batch_size:this_batch_size * 2, :, :, :]

        # concat cover and original secret get the concat_img with 6 channels
        concat_img = torch.cat([cover_img, secret_img], dim=1)

        # data into GPU
        if opt.cuda:
            cover_img = cover_img.cuda()
            secret_img = secret_img.cuda()
            concat_img = concat_img.cuda()

        concat_imgv = Variable(concat_img, volatile=True)  # concat_img is the input of Hiding net
        cover_imgv = Variable(cover_img, volatile=True)  # cover_imgv is the label of Hiding net

        container_img = Hnet(concat_imgv)  # concat_img as the input of HidingNet and get the container_img
        errH = criterion(container_img, cover_imgv)  # Hiding net reconstructed error
        Hlosses.update(errH.data[0], this_batch_size)  # record the H loss value

        rev_secret_img = Rnet(
            container_img)  # containerImg is the input of the Rnet and get the output "rev_secret_img"
        secret_imgv = Variable(secret_img, volatile=True)  # secret_imgv is the label of Rnet
        errR = criterion(rev_secret_img, secret_imgv)  # Reveal net reconstructed error
        Rlosses.update(errR.data[0], this_batch_size)  # record the R loss value
        save_result_pic(this_batch_size, cover_img, container_img.data, secret_img, rev_secret_img.data, epoch, i,
                        opt.testPics)

    val_hloss = Hlosses.avg
    val_rloss = Rlosses.avg
    val_sumloss = val_hloss + opt.beta * val_rloss

    val_time = time.time() - start_time
    val_log = "validation[%d] val_Hloss = %.6f\t val_Rloss = %.6f\t val_Sumloss = %.6f\t validation time=%.2f" % (
        epoch, val_hloss, val_rloss, val_sumloss, val_time)
    print_log(val_log, logPath)

    print(
        "#################################################### test end ########################################################")
    return val_hloss, val_rloss, val_sumloss


# print the training log and save into logFiles
def print_log(log_info, log_path, console=True):
    # print the info into the console
    if console:
        print(log_info)
    # debug mode don't write the log into files
    if not opt.debug:
        # write the log into log file
        if not os.path.exists(log_path):
            fp = open(log_path, "w")
            fp.writelines(log_info + "\n")
        else:
            with open(log_path, 'a+') as f:
                f.writelines(log_info + '\n')


# save result pic and the coverImg filePath and the secretImg filePath
def save_result_pic(this_batch_size, originalLabelv, ContainerImg, secretLabelv, RevSecImg, epoch, i, save_path):
    if not opt.debug:
        originalFrames = originalLabelv.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
        containerFrames = ContainerImg.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
        secretFrames = secretLabelv.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)
        revSecFrames = RevSecImg.resize_(this_batch_size, 3, opt.imageSize, opt.imageSize)

        showContainer = torch.cat([originalFrames, containerFrames], 0)
        showReveal = torch.cat([secretFrames, revSecFrames], 0)
        # resultImg contains four rows，each row is coverImg containerImg secretImg RevSecImg, total this_batch_size columns
        resultImg = torch.cat([showContainer, showReveal], 0)
        resultImgName = '%s/ResultPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)
        vutils.save_image(resultImg, resultImgName, nrow=this_batch_size, padding=1, normalize=True)


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
