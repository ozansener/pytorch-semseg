import sys
import torch
import click
import datetime
import json
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import runningScore
from ptsemseg.loss import *
from ptsemseg.augmentations import *

from tensorboardX import SummaryWriter


@click.command()
@click.option('--param_file', default='params.json', help='JSON parameters file')
def train(param_file):
    with open(param_file) as json_params:
        params = json.load(json_params)
    exp_identifier = '|'.join('{}={}'.format(key,val) for (key,val) in params.items())
    params['exp_id'] = exp_identifier

    # Setup Augmentations
    data_aug= Compose([RandomRotate(10),
                       RandomHorizontallyFlip()])

    # Setup Dataloader
    data_loader = get_loader(params['dataset'])
    data_path = get_data_path(params['dataset'])
    t_loader = data_loader(data_path, is_transform=True, split=['train'],  img_size=(params['img_rows'], params['img_cols']), augmentations=data_aug)
    v_loader = data_loader(data_path, is_transform=True, split=['val'], img_size=(params['img_rows'], params['img_cols']))

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=params['batch_size'], num_workers=8, shuffle=True)
    valloader = data.DataLoader(v_loader, batch_size=params['batch_size'], num_workers=8)

    # Setup Metrics
    running_metrics = runningScore(n_classes)

    writer = SummaryWriter(log_dir='runs/{}_{}'.format(params['exp_id'], datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")))

    # Setup Model
    model = get_model(params['arch'], n_classes, params['tasks'])

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()

    print(params)
    if 'RMSprop' in params['optimizer']:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=params['lr'])
    elif 'Adam' in params['optimizer']:
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    elif 'SGD' in params['optimizer']:
        optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=0.9)


    loss_fn = {}
    if hasattr(model.module, 'loss'):
        print('Using custom loss')
        loss_fn = model.module.loss
    else:
        loss_fn['S'] = cross_entropy2d
        loss_fn['I'] = l1_loss_instance
    """
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading model and optimizer from checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))
    """

    tasks = []
    if 'S' in params['tasks']:
        tasks.append('S')
    if 'I' in params['tasks']:
        tasks.append('I')

    best_iou = -100.0
    best_loss = 1e8
    n_iter = 0
    for epoch in range(100):
        model.train()
        if (epoch+1) % 30 == 0:
            # Every 10 epoch, half the LR
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print('Half the learning rate{}'.format(n_iter))
        target = {}
        for i, (images, labels, instances, imname) in enumerate(trainloader):
            n_iter += 1
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            instances = Variable(instances.cuda())

            target['S'] = labels
            target['I'] = instances
 
            optimizer.zero_grad()
            outputs = model(images)

            for task_id, task in enumerate(tasks):
                if task_id > 0:
                    loss = loss + loss_fn[task](input=outputs[task_id], target= target[task])
                else:
                    loss = loss_fn[task](input = outputs[0], target = target[task])


            if loss is None:
                print('WARN: image with no instance {}'.format(imname))
                continue

            loss.backward()
            optimizer.step()
            writer.add_scalar('training_loss', loss.data[0], n_iter)
        model.eval()
        tot_loss = 0.0
        summed = 0.0
        target_val = {}
        val_losses = {}
        for task in tasks:
            val_losses[task] = 0.0
        for i_val, (images_val, labels_val, instances_val, imname_val) in enumerate(valloader):
            images_val = Variable(images_val.cuda(), volatile=True)
            labels_val = Variable(labels_val.cuda(), volatile=True)
            instances_val = Variable(instances_val.cuda(), volatile=True)

            target_val['S'] = labels_val
            target_val['I'] = instances_val
 
            outputs = model(images_val)

            for task_id, task in enumerate(tasks):
                if task_id > 0:
                    ll =  loss_fn[task](input=outputs[task_id],target= target_val[task])
                    val_losses[task] += ll.data[0]
                else:
                    ll = loss_fn[task](input = outputs[0], target = target_val[task])
                    val_losses[task] += ll.data[0]
                if 'S' in task:
                    pred_cpu = outputs[task_id].data.max(1)[1].cpu().numpy()
                    gt_cpu = labels_val.data.cpu().numpy()
                    running_metrics.update(gt_cpu, pred_cpu)
            summed += 1
            #running_metrics.update_instance(instances_gt, outputs.data.cpu().numpy())
        for task in tasks:
            writer.add_scalar('validation_loss_{}'.format(task), val_losses[task]/summed, n_iter)
        writer.add_scalar('validation_loss_total', sum(val_losses.values())/summed, n_iter)

        score, class_iou = running_metrics.get_scores()
        for k, v in score.items():
            writer.add_scalar('score_{}'.format(k), v, n_iter)
        running_metrics.reset()

        """
        if tot_loss <= best_loss:
            best_loss = tot_loss
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),}
            torch.save(state, "saved_models/{}_{}_model.pkl".format(params['exp_id'], tot_loss))"""

if __name__ == '__main__':
    train()
