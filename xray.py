import os
import argparse
import csv
import pandas as pd
import importlib
import torch
from torch.autograd import Variable
import torch.nn as nn
import sys
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from models.models import classifier32, classifier32ABN, Generator, Discriminator, classifier64ABN
from datasets.osr_dataloader import MNIST_OSR, CIFAR10_OSR, CIFAR100_OSR, SVHN_OSR, Tiny_ImageNet_OSR
from datasets.datasets_med import build_dataset_chest_xray
from torch.utils.tensorboard import SummaryWriter
from split import splits_AUROC, splits_F1
from sklearn.metrics import f1_score
import os.path as osp
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm
from densenet import DenseNet

parser = argparse.ArgumentParser("Training")
parser.add_argument('--dataset', type=str, default='chestxray', help="mnist | svhn | cifar10 | cifar100 | tiny_imagenet")
parser.add_argument('--dataroot', type=str, default='./data')
parser.add_argument('--out-num', type=int, default=50, help='For CIFAR100')
parser.add_argument('--optimizer', type=str, default='SGD')
parser.add_argument('--batch-size', type=int, default=32) # epoch 100 -> 32
parser.add_argument('--lr', type=float, default=0.001, help="learning rate for model")
parser.add_argument('--gan_lr', type=float, default=0.0002, help="learning rate for gan")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--img_size', type=int, default=32)
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--fake_ratio', type=float, default=0.1)
parser.add_argument('--pull_ratio', type=float, default=1.0)
parser.add_argument('--smoothing', type=float, default=0.5)
parser.add_argument('--smoothing2', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save-dir', type=str, default='../log')
parser.add_argument('--split', type=str, default='F1', help="F1 | AUROC")
# class SmoothCrossEntropy(nn.Module):
#     def __init__(self, alpha=0.5):
#         super(SmoothCrossEntropy, self).__init__()
#         self.alpha = alpha
#
#     def forward(self, logits, labels, loss):
#         num_classes = logits.shape[-1]
#         alpha_div_k = self.alpha / num_classes
#         # target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
#         #     (1. - self.alpha) + alpha_div_k
#         target_probs = labels.float() * (1. - self.alpha) + alpha_div_k
#         # loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
#         # return loss.mean()
#         return loss(logits, target_probs)
to_np = lambda x: x.data.cpu().numpy()

class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.5):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels, loss):
        num_classes = logits.shape[-1]
        # alpha_div_k = self.alpha / num_classes
        # target_probs = labels.float() * (1. - self.alpha) + alpha_div_k
        alpha_div_k = 0.4
        target_probs = labels.float() * (1. - self.alpha) + alpha_div_k
        return loss(logits, target_probs)


def jointEnergy(logits):
    E_f = torch.log(1+torch.exp(logits))
    scores = torch.sum(E_f, dim=1)
    return scores

def train(net1, net2, criterion, optimizer1, optimizer2, trainloader, **options):
    lsr_criterion = SmoothCrossEntropy(options['smoothing'])
    lsr_criterion2 = SmoothCrossEntropy(options['smoothing2'])
    l1_loss = nn.L1Loss()

    torch.cuda.empty_cache()

    for batch_idx, (data, labels) in enumerate(tqdm(trainloader)):
        data, labels = data.cuda(), labels.cuda()
        bsz = labels.size(0)
        with torch.set_grad_enabled(True):
            # net1.train()
            net2.eval()
            optimizer1.zero_grad()
            feat11, feat12, feat13, feat1, out1 = net1(data, return_feature=True, layers=[1,2,3])
            feat21, feat22, feat23, feat2, out2 = net2(data, return_feature=True, layers=[1,2,3])
            loss1 = criterion(out1, labels)
            # entropy = torch.sum((-torch.sigmoid(out1) * torch.log2(torch.sigmoid(out1))), dim=1)
            # entropy = torch.mean(entropy)
            # loss1 += 0.1 * entropy
            pullloss1 = l1_loss(feat11.reshape(bsz, -1), feat21.reshape(bsz, -1).detach())#,options['margin'])
            pullloss2 = l1_loss(feat12.reshape(bsz, -1), feat22.reshape(bsz, -1).detach())#,options['margin'])
            pullloss3 = l1_loss(feat13.reshape(bsz, -1), feat23.reshape(bsz, -1).detach())#, options['margin'])
            # pullloss = (pullloss1 + pullloss2) / 2
            pullloss = (pullloss1 + pullloss2 + pullloss3) / 3
            loss1 = loss1 + options['pull_ratio'] * pullloss
            loss1.backward()
            optimizer1.step()

            net1.eval()
            net2.train()
            optimizer2.zero_grad()
            feat11, feat12, feat13, feat1, out1 = net1(data, return_feature=True, layers=[1,2,3])
            feat21, feat22, feat23, feat2, out2 = net2(data, return_feature=True, layers=[1,2,3])
            # out2 = net2(data)
            out21 = net2(feat11.detach(), input_layers=1)
            out22 = net2(feat12.detach(), input_layers=2)
            out23 = net2(feat13.detach(), input_layers=3)
            out20 = net2(feat1.clone().detach(), onlyfc=True)
            klu0 = lsr_criterion2(out20, labels, criterion)
            klu1 = lsr_criterion(out21, labels, criterion)
            klu2 = lsr_criterion(out22, labels, criterion)
            klu3 = lsr_criterion(out23, labels, criterion)
            # klu = (klu0 + klu1 + klu2) / 3
            klu = (klu0 + klu1 + klu2 + klu3) / 4
            loss2 = criterion(out2, labels)
            loss2 = loss2 + klu * options['fake_ratio']

            # loss2 = loss2 + 0.1 * (l2_loss(entropy1, entropy0) + l2_loss(entropy2, entropy0) + l2_loss(entropy3, entropy0))/3
            loss2.backward()
            # print(loss2.data)
            optimizer2.step()

            net2.train()
            optimizer2.zero_grad()
            out2 = net2(data, return_feature=False)
            loss2 = criterion(out2, labels)
            loss2.backward()
            optimizer2.step()


def train_gan(net, netD, netG, criterion, criterionD, optimizer2, optimizerD, optimizerG, trainloader, **options):
    net.train()
    netD.train()
    netG.train()
    torch.cuda.empty_cache()
    lsr_criterion = SmoothCrossEntropy(options['smoothing2'])
    real_label, fake_label = 1, 0
    for batch_idx, (data, labels) in enumerate(trainloader):
        gan_target = torch.FloatTensor(labels.size()).fill_(0)
        noise = torch.FloatTensor(data.size(0), 100, 1, 1).normal_(0, 1).cuda()
        data, labels, gan_target, noise = data.cuda(non_blocking=True), labels.cuda(non_blocking=True), gan_target.cuda(), noise.cuda()

        fake = netG(noise)
        gan_target.fill_(real_label)
        targetv = Variable(gan_target)
        optimizerD.zero_grad()
        output = netD(data)
        errD_real = criterionD(output, targetv)
        errD_real.backward()

        targetv = Variable(gan_target.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterionD(output, targetv)
        errD_fake.backward()
        optimizerD.step()

        optimizerG.zero_grad()
        targetv = Variable(gan_target.fill_(real_label))
        output = netD(fake)
        errG = criterionD(output, targetv)
        x = net(fake, bn_label=1 * torch.ones(data.shape[0], dtype=torch.long).cuda())
        errG_F = lsr_criterion(x, labels)
        generator_loss = errG + options['beta'] * errG_F * -1.
        generator_loss.backward()
        optimizerG.step()

        optimizer2.zero_grad()
        x = net(data,bn_label=0 * torch.ones(data.shape[0], dtype=torch.long).cuda())
        loss = criterion(x, labels)
        noise = torch.FloatTensor(data.size(0), 100, 1, 1).normal_(0, 1).cuda()
        noise = noise.cuda()
        noise = Variable(noise)
        fake = netG(noise)
        x= net(fake, bn_label=1 * torch.ones(data.shape[0], dtype=torch.long).cuda())
        F_loss_fake = lsr_criterion(x, labels)
        total_loss = loss + options['beta'] * F_loss_fake
        total_loss.backward()
        optimizer2.step()


def evaluation(net2, testloader, outloader, **options):
    net2.eval()
    correct, total, n = 0, 0, 0
    torch.cuda.empty_cache()
    pred_close, pred_open, labels_close, labels_open = [], [], [], []
    open_labels = torch.zeros(50000)
    probs = torch.zeros(50000)
    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.cuda(), labels.cuda()
            bsz = labels.size(0)
            with torch.set_grad_enabled(False):
                _, logits = net2(data, return_feature=True)
                logits = torch.softmax(logits / options['temp'], dim=1)
                confidence = logits.data.max(1)[0]
                for b in range(bsz):
                    probs[n] = confidence[b]
                    open_labels[n] = 1
                    n += 1
                predictions = logits.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()
                pred_close.append(logits.data.cpu().numpy())
                labels_close.append(labels.data.cpu().numpy())

        for batch_idx, (data, labels) in enumerate(outloader):
            data, labels = data.cuda(), labels.cuda()
            bsz = labels.size(0)
            oodlabel = torch.zeros_like(labels) - 1
            with torch.set_grad_enabled(False):
                _, logits = net2(data, return_feature=True)
                logits = torch.sigmoid(logits)
                confidence = logits.data.max(1)[0]
                for b in range(bsz):
                    probs[n] = confidence[b]
                    open_labels[n] = 0
                    n += 1
                pred_open.append(logits.data.cpu().numpy())
                labels_open.append(oodlabel.data.cpu().numpy())
    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

    pred_close = np.concatenate(pred_close, 0)
    pred_open = np.concatenate(pred_open, 0)
    labels_close = np.concatenate(labels_close, 0)
    labels_open = np.concatenate(labels_open, 0)
    # F1 score Evaluation
    x1, x2 = np.max(pred_close, axis=1), np.max(pred_open, axis=1)
    pred1, pred2 = np.argmax(pred_close, axis=1), np.argmax(pred_open, axis=1)
    total_pred_label = np.concatenate([pred1, pred2], axis=0)
    total_label = np.concatenate([labels_close, labels_open], axis=0)
    total_pred = np.concatenate([x1, x2], axis=0)
    thr = options['smoothing'] / options['num_classes'] + (1 - options['smoothing'])
    open_pred = (total_pred > thr - 0.05).astype(np.float32)
    f = f1_score(total_label, ((total_pred_label + 1) * open_pred) - 1, average='macro')

    # AUROC score Evaluation
    open_labels = open_labels[:n].cpu().numpy()
    prob = probs[:n].reshape(-1, 1)
    auc = roc_auc_score(open_labels, prob)

    return acc, auc, f

def computeAUROC(dataGT, dataPRED, classCount):
    outAUROC = []
    # print(dataGT.shape, dataPRED.shape)
    for i in range(classCount):
        try:
            outAUROC.append(roc_auc_score(dataGT[:, i], dataPRED[:, i]))
        except:
            outAUROC.append(0.)
    return outAUROC

def evaluation2(net2, validloader, **options):
    net2.eval()
    torch.cuda.empty_cache()
    pred_close, pred_open, labels_close, labels_open = [], [], [], []
    with torch.no_grad():
        for data, labels in validloader:
            data, labels = data.cuda(), labels.cuda()
            with torch.set_grad_enabled(False):
                _, logits = net2(data, return_feature=True)
                # scores = jointEnergy(logits)
                logits = torch.sigmoid(logits)
                pred_close.append(logits.data.cpu().numpy())
                labels_close.append(labels.data.cpu().numpy())
                # labels_open.append(osrs.data.cpu().numpy())
                # pred_open.append(-to_np(scores))

    pred_close = np.concatenate(pred_close, 0)
    # pred_open = np.concatenate(pred_open, 0)
    labels_close = np.concatenate(labels_close, 0)
    # labels_open = np.concatenate(labels_open, 0)

    auc_close = computeAUROC(labels_close, pred_close, 14)
    # auc_open = roc_auc_score(labels_open, pred_open)

    return auc_close#, auc_open

def evaluation3(net2, validloader, **options):
    net2.eval()
    torch.cuda.empty_cache()
    pred_close, pred_open, labels_close, labels_open = [], [], [], []
    with torch.no_grad():
        for data, labels in validloader:
            data, labels = data.cuda(), labels.cuda()
            with torch.set_grad_enabled(False):
                _, logits = net2(data, return_feature=True)
                scores = jointEnergy(logits)
                labels_open.append(labels.data.cpu().numpy())
                pred_open.append(to_np(scores))

    pred_open = np.concatenate(pred_open, 0)
    labels_open = np.concatenate(labels_open, 0)


    auc_open = roc_auc_score(labels_open, pred_open)

    return auc_open

def main(options):
    torch.manual_seed(options['seed'])
    use_gpu = torch.cuda.is_available()
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(options['seed'])


    # Dataset
    print("{} Preparation".format(options['dataset']))
    if 'mnist' in options['dataset']:
        Data = MNIST_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'],
                         img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'cifar10' == options['dataset']:
        Data = CIFAR10_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'],
                           img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'svhn' in options['dataset']:
        Data = SVHN_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'],
                        img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'cifar100' in options['dataset']:
        Data = CIFAR10_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'],
                           img_size=options['img_size'])
        trainloader, testloader = Data.train_loader, Data.test_loader
        out_Data = CIFAR100_OSR(known=options['unknown'], dataroot=options['dataroot'],
                                batch_size=options['batch_size'], img_size=options['img_size'])
        outloader = out_Data.test_loader
    elif 'chestxray' in options['dataset']:
        dataset = options['dataset']
        dataset_train = build_dataset_chest_xray(split='train', dataset=dataset, file='./data_splits/chestxray/train_official.txt')
        dataset_val = build_dataset_chest_xray(split='val', dataset=dataset, file='./data_splits/chestxray/val_official.txt')
        dataset_test = build_dataset_chest_xray(split='test', dataset=dataset, file='./data_splits/chestxray/test_official.txt')
        dataset_covid = build_dataset_chest_xray(split='test', dataset='covidx', file='./data/COVIDx-CXR3')

        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        sampler_covid = torch.utils.data.SequentialSampler(dataset_covid)

        trainloader = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=options['batch_size'],
            num_workers=4,
            pin_memory='store_true',
            drop_last=True,
        )

        testloader = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=256,
            num_workers=4,
            pin_memory='store_true',
            drop_last=False
        )

        validloader = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=256,
            num_workers=4,
            pin_memory='store_true',
            drop_last=False
        )

        covidloader = torch.utils.data.DataLoader(
            dataset_covid, sampler=sampler_covid,
            batch_size=256,
            num_workers=4,
            pin_memory='store_true',
            drop_last=False
        )


    else:
        Data = Tiny_ImageNet_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'],
                                 img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    options['num_classes'] = 14
    net1 = DenseNet(num_classes=options['num_classes'])
    net2 = DenseNet(num_classes=options['num_classes'])
    net1 = nn.DataParallel(net1).cuda()
    criterion = nn.BCEWithLogitsLoss()
    net2 = nn.DataParallel(net2).cuda()
    criterion = criterion.cuda()

    if options['optimizer'] == 'adam':
        options['lr'] = 0.001
        optimizer1 = torch.optim.Adam(net1.parameters(), lr=options['lr'])
        optimizer2 = torch.optim.Adam(net2.parameters(), lr=options['lr'])
        scheduler1 = lr_scheduler.MultiStepLR(optimizer1, milestones=[options['max_epoch'] // 2])
        scheduler2 = lr_scheduler.MultiStepLR(optimizer2, milestones=[options['max_epoch'] // 2])
    else:
        options['lr'] = 0.1
        optimizer1 = torch.optim.SGD(net1.parameters(), lr=options['lr'], momentum=0.9, weight_decay=1e-4)
        optimizer2 = torch.optim.SGD(net2.parameters(), lr=options['lr'], momentum=0.9, weight_decay=1e-4)
        scheduler1 = lr_scheduler.MultiStepLR(optimizer1, milestones=[60, 120, 180, 240])
        scheduler2 = lr_scheduler.MultiStepLR(optimizer2, milestones=[60, 120, 180, 240])

    best_open = 0
    best_close = 0

    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch + 1, options['max_epoch']))
        # train_gan(net2, netD, netG, criterion, criterionD, optimizer2, optimizerD, optimizerG, trainloader, **options)
        train(net1, net2, criterion, optimizer1=optimizer1, optimizer2=optimizer2, trainloader=trainloader, **options)

        scheduler1.step()
        scheduler2.step()

        val_close = evaluation2(net2, validloader, **options)
        val_close = np.mean(val_close)
        print('val_close:', val_close)
        if val_close > best_close:
            best_close = val_close
            test_close = evaluation2(net2, testloader, **options)
            test_open = evaluation3(net2, covidloader, **options)
            print(test_open, np.mean(test_close))
            stats_log = open('./logs/' + options['split'] + '/' + options['dataset'] + '/DIAS_%d' % (i) + '.txt', 'a')
            stats_log.write("Epoch, Open, Close:" + str(epoch) + ',' +  str(test_open) + ',' + str(np.mean(test_close)) + ',' + str(test_close) + '\n')
            stats_log.flush()
            stats_log.close()
            save_checkpoint(epoch, {
                'epoch': epoch,
                'net2_state_dict': net2.state_dict(),
                'optimizer2': optimizer2.state_dict(),
            }, options['batch_size'], options['item'], True)
        # elif val_open > best_open:
        #     best_open = val_open
        #     test_close, test_open = evaluation2(net2, testloader, **options)
        #     stats_log = open('./logs/' + options['split'] + '/' + options['dataset'] + '/DIAS_%d' % (i) + '.txt', 'a')
        #     stats_log.write("Epoch, Open, Close:" + str(epoch) + ',' + str(test_open) + ',' + str(np.mean(test_close)) + ',' + str(test_close) + '\n')
        #     stats_log.flush()
        #     stats_log.close()
        #     save_checkpoint(epoch, {
        #         'epoch': epoch,
        #         'net2_state_dict': net2.state_dict(),
        #         'optimizer2': optimizer2.state_dict(),
        #     }, options['batch_size'], options['item'], True)
    else:
        test_close = evaluation2(net2, testloader, **options)
        test_open = evaluation3(net2, covidloader, **options)
        stats_log = open('./logs/' + options['split'] + '/' + options['dataset'] + '/DIAS_%d' % (i) + '.txt', 'a')
        stats_log.write("Epoch, Open, Close:" + str(epoch) + ',' + str(test_open) + ',' + str(np.mean(test_close)) + ',' + str(test_close) + '\n')
        stats_log.write("-------------------------")
        stats_log.flush()
        stats_log.close()
        save_checkpoint(epoch, {
            'epoch': epoch,
            'net2_state_dict': net2.state_dict(),
            'optimizer2': optimizer2.state_dict(),
        }, options['batch_size'], options['item'], True)
    return test_close, test_open

def save_checkpoint(epoch, state, bsz, item, is_best=True, filename='checkpoint.pth.tar'):
    directory = "runs/%s/%s/" % (str(options['split']), options['dataset'])
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + str(item) + '_' + str(epoch) + '_' + str(bsz) + '_' + filename
    torch.save(state, filename)

if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)
    if options['split'] == 'AUROC':
        splits = splits_AUROC
    elif options['split'] == 'F1':
        splits = splits_F1
    else:
        raise NotImplementedError()
    options['dataroot'] = os.path.join(options['dataroot'], options['dataset'])

    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./logs/AUROC'):
        os.makedirs('./logs/AUROC')
    if not os.path.exists('./logs/F1'):
        os.makedirs('./logs/F1')
    if not os.path.exists('./logs/AUROC/' + options['dataset']):
        os.makedirs('./logs/AUROC/' + options['dataset'])
    if not os.path.exists('./logs/F1/' + options['dataset']):
        os.makedirs('./logs/F1/' + options['dataset'])
    for i in range(len(splits[options['dataset']])):
        options['item'] = i
        options['writer'] = SummaryWriter(f"results/DIAS_{i}")
        known = splits[options['dataset']][len(splits[options['dataset']]) - i - 1]
        if options['dataset'] == 'cifar100':
            unknown = splits[options['dataset'] + '-' + str(options['out_num'])][
                len(splits[options['dataset']]) - i - 1]
        elif options['dataset'] == 'tiny_imagenet':
            options['lr'] = 0.001
            unknown = list(set(list(range(0, 200))) - set(known))
        else:
            unknown = list(set(list(range(0, 10))) - set(known))
        options.update({'known': known, 'unknown': unknown})
        if options['optimizer'] == 'adam':
            options['lr'] = 0.001
        else:
            options['lr'] = 0.1
        stats_log = open('./logs/' + options['split'] + '/' + options['dataset'] + '/DIAS_%d' % (i) + '.txt', 'w')
        stats_log.close()
        best_close, best_open = main(options)