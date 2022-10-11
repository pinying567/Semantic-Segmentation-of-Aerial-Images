import numpy as np
import argparse
import os
import json
import torch
import torch.utils.data as data
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
from PIL import Image

from util import out_mask, averageMeter, lr_decay
from dataset import Dataset
from fcn32s import vgg16_fcn32s

img_size = 512

def main():
    global save_dir, logger
    
    # setup directory to save logfiles, checkpoints, and output
    save_dir = args.save_dir
    if args.phase == 'train' and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # setup logger
    logger = None
    if args.phase == 'train':
        logger = open(os.path.join(save_dir, 'train.log'), 'a')
        logfile = os.path.join(save_dir, 'training_log.json')
        log = {'train': [], 'val': []}
        logger.write('{}\n'.format(args))
    
    # setup data loader for training images
    if args.phase == 'train':
        trans_train = transforms.Compose([
            transforms.Resize(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tgt_trans_train = transforms.Compose([
            transforms.Resize(args.img_size, interpolation=0),
            transforms.ToTensor(),
        ])
        
        dataset_train = Dataset(os.path.join(args.data_root, 'p2_data/train'), trans_train, tgt_trans_train)
        train_loader = data.DataLoader(dataset_train, shuffle=True, drop_last=False, pin_memory=True, batch_size=args.batch_size)
        print('train: {}'.format(dataset_train.__len__()))
        logger.write('train: {}\n'.format(dataset_train.__len__()))
    
    # setup data loader for validation/testing images
    trans_val = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tgt_trans_val = transforms.Compose([
        transforms.Resize(args.img_size, interpolation=0),
        transforms.ToTensor(),
    ])
    
    if args.test_dir:
        dataset_val = Dataset(args.test_dir, trans_val, tgt_trans_val)
    else:
        dataset_val = Dataset(os.path.join(args.data_root, 'p2_data/validation'), trans_val, tgt_trans_val)
    
    print('val/test: {}'.format(dataset_val.__len__()))
    if args.phase == 'train':
        logger.write('val/test: {}\n'.format(dataset_val.__len__()))
        
    val_loader = data.DataLoader(dataset_val, shuffle=False, drop_last=False, batch_size=args.batch_size)

    # setup model (load pretrained model)    
    if args.phase == 'train':
        model = vgg16_fcn32s(pretrained=True, **{'num_classes': 7}).cuda()
        logger.write('{}\n'.format(model))
    else:
        model = vgg16_fcn32s(pretrained=False, **{'num_classes': 7}).cuda()
    
    # setup loss function
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    # setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.phase == 'train':
        logger.write('{}\n'.format(optimizer))   
    
    # load checkpoint
    start_ep = 0
    best_mIOU = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['opt_state'])
        start_ep = checkpoint['epoch']
        best_mIOU = checkpoint['mIOU']
        print("Loaded checkpoint '{}' (epoch: {}, mIOU: {})".format(args.checkpoint, start_ep, best_mIOU))
        
        if args.phase == 'train':
            logger.write("Loaded checkpoint '{}' (epoch: {}, mIOU: {})\n".format(args.checkpoint, start_ep, best_mIOU))
            if os.path.isfile(logfile):
                log = json.load(open(logfile, 'r'))

    if args.phase == 'train':       
        
        # start training
        print('Start training from epoch {}'.format(start_ep))
        logger.write('Start training from epoch {}\n'.format(start_ep))
        for epoch in range(start_ep, args.epochs):
            
            mIOU_train, loss_train = train(train_loader, model, optimizer, epoch, criterion)
            log['train'].append([epoch + 1, mIOU_train, loss_train])
            
            if (epoch + 1) % args.val_ep == 0:
                with torch.no_grad():
                    mIOU_val, loss_val = val(val_loader, model, criterion)
                log['val'].append([epoch + 1, mIOU_val, loss_val])
                
                # save checkpoint
                state = {
                    'epoch': epoch + 1,
                    'mIOU': mIOU_val,
                    'model_state': model.state_dict(),
                    'opt_state': optimizer.state_dict()
                }
                checkpoint = os.path.join(save_dir, 'ep-{}.pkl'.format(epoch + 1))
                torch.save(state, checkpoint)
                print('[Checkpoint] {} is saved.'.format(checkpoint))
                logger.write('[Checkpoint] {} is saved.\n'.format(checkpoint))
                json.dump(log, open(logfile, 'w'))
                
                if mIOU_val > best_mIOU:
                    # save checkpoint
                    state = {
                        'epoch': epoch + 1,
                        'mIOU': mIOU_val,
                        'model_state': model.state_dict(),
                        'opt_state': optimizer.state_dict()
                    }
                    checkpoint = os.path.join(save_dir, 'best_model.pkl')
                    torch.save(state, checkpoint)
                    print('[Checkpoint] {} is saved.'.format(checkpoint))
                    logger.write('[Checkpoint] {} is saved.\n'.format(checkpoint))
                    json.dump(log, open(logfile, 'w'))
                    best_mIOU = mIOU_val
                    
            if (epoch + 1) % args.step == 0:
                lr_decay(optimizer, decay_rate=args.gamma)
        
        # save last model
        state = {
            'epoch': epoch + 1,
            'mIOU': mIOU_val,
            'model_state': model.state_dict(),
            'opt_state': optimizer.state_dict()
        }
        checkpoint = os.path.join(save_dir, 'last_checkpoint.pkl')
        torch.save(state, checkpoint)
        print('[Checkpoint] {} is saved.'.format(checkpoint))
        logger.write('[Checkpoint] {} is saved.\n'.format(checkpoint))
        print('Training is done.')
        logger.write('Training is done.\n')
        logger.close()
        
    else:
        with torch.no_grad():
            mIOU_val, loss_val = val(val_loader, model, criterion, save_result=True)
            
        print('Testing is done.')
        

def train(data_loader, model, optimizer, epoch, criterion):

    losses = averageMeter()
    TP_FP = torch.zeros([6]).cuda()
    TP_FN = torch.zeros([6]).cuda()
    TP = torch.zeros([6]).cuda()
    
    # setup training mode
    model.train()
    
    for (step, value) in enumerate(data_loader):

        image = value[0].cuda()
        target = value[2].cuda(non_blocking=True)
            
        # forward
        output = model(image)

        # compute loss
        loss = torch.mean(criterion(output, target).squeeze())
        losses.update(loss.item(), image.size(0))
            
        # compute mIOU
        with torch.no_grad():
            pred = torch.max(torch.softmax(output, dim=1), dim=1)[1]
            for i in range(6):
                pred_i = (pred == i).float()
                labels_i = (target == i).float()
                TP_FP[i] += torch.sum(pred_i)
                TP_FN[i] += torch.sum(labels_i)
                TP[i] += torch.sum(pred_i * labels_i)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        mIOU = 0
        for i in range(6):
            iou = TP[i] / (TP_FP[i] + TP_FN[i] - TP[i])
            mIOU += iou / 6
    
    # logging
    curr_lr = optimizer.param_groups[0]['lr']
    print('Epoch: [{}/{}]\t' \
        'LR: [{:.6g}]\t' \
        'Loss {loss.avg:.4f}\t' \
        'mIOU {mIOU:.3f}'.format(
            epoch + 1, args.epochs, curr_lr, loss=losses, mIOU=mIOU
        )
    )
    logger.write('Epoch: [{}/{}]\t' \
        'LR: [{:.6g}]\t' \
        'Loss {loss.avg:.4f}\t' \
        'mIOU {mIOU:.3f}\n'.format(
            epoch + 1, args.epochs, curr_lr, loss=losses, mIOU=mIOU
        )
    )
    return mIOU.item(), losses.avg

    
def val(data_loader, model, criterion, save_result=False):

    losses = averageMeter()
    TP_FP = torch.zeros([6]).cuda()
    TP_FN = torch.zeros([6]).cuda()
    TP = torch.zeros([6]).cuda()
    
    # setup evaluation mode
    model.eval()
    
    for (step, value) in enumerate(data_loader):

        image = value[0].cuda()
        target = None
        if len(value) > 2:
            target = value[2].cuda(async=True)
        
        # forward
        output = model(image)
        
        # save image_id & predictions
        if save_result:
            pred_batch = torch.max(output, dim=1)[1].data.cpu().numpy()
            for i in range(image.size(0)):
                output_img = out_mask(pred_batch[i])
                fname = os.path.join(save_dir, value[1][i].replace('sat.jpg', 'mask.png'))
                Image.fromarray(output_img).resize((img_size, img_size), resample=0).save(fname)
        
        # compute loss
        if target is not None:
            loss = torch.mean(criterion(output, target).squeeze())
            losses.update(loss.item(), image.size(0))
            
            # compute mIOU
            with torch.no_grad():
                pred = torch.max(torch.softmax(output, dim=1), dim=1)[1]
                for i in range(6):
                    pred_i = (pred == i).float()
                    labels_i = (target == i).float()
                    TP_FP[i] += torch.sum(pred_i)
                    TP_FN[i] += torch.sum(labels_i)
                    TP[i] += torch.sum(pred_i * labels_i)
    
    with torch.no_grad():
        mIOU = 0
        for i in range(6):
            iou = TP[i] / (TP_FP[i] + TP_FN[i] - TP[i])
            mIOU += iou / 6
    
    if losses.count > 0:
        # logging
        print('[Val] Loss {loss.avg:.4f}\tmIOU {mIOU:.3f}'.format(loss=losses, mIOU=mIOU))
        if args.phase == 'train':
            logger.write('[Val] Loss {loss.avg:.4f}\tmIOU {mIOU:.3f}'.format(loss=losses, mIOU=mIOU))
    
    return mIOU.item(), losses.avg


if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size (default: 16)')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.02, help='base learning rate (default: 0.02)')
    parser.add_argument('--step', type=int, default=10, help='learning rate decay step (default: 10)')
    parser.add_argument('--val_ep', type=int, default=5, help='validation period (default: 5)')
    parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate step gamma (default: 0.1)')
    parser.add_argument('--test_dir', type=str, default='', help='testing image directory')
    parser.add_argument('--checkpoint', type=str, default='', help='pretrained model')
    parser.add_argument('--save_dir', type=str, default='checkpoint/vgg16_fcn32s', help='directory to save logfile, checkpoint and output segmetation masks')
    parser.add_argument('--data_root', type=str, default='data', help='data root')
    parser.add_argument('--phase', type=str, default='train', help='phase (train/test)')
    parser.add_argument('--img_size', type=int, default=224, help='image size (default: 224)')

    args = parser.parse_args()
    print(args)
    
    main()