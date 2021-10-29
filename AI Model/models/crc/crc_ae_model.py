"""
Created by Wang Han on 2019/3/29 11:29.
E-mail address is hanwang.0501@gmail.com.
Copyright © 2019 Wang Han. SCU. All Rights Reserved.
"""

import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

import datasets.crc.transforms.crc_transforms_3d_ae as CRCT3d
from datasets.crc.crc_dataset_3d_ae import CRCDataset3DAE
from nets.ae_net_selector import AENetSelector
from nets.cores.aeloss import AELoss
from nets.cores.meter import AverageMeter


class Model:

    def __init__(self, config):
        self.config = config
        # loading network parameters
        network_params = config['network']
        self.device = torch.device(network_params['device'])
        self.net = AENetSelector(config).get_net()

        if network_params['use_parallel']:
            self.net = nn.DataParallel(self.net)

        self.net = self.net.to(self.device)
        self.epochs = config['optim']['num_epochs']

        # loading logging parameters
        logging_params = config['logging']
        run_timestamp = datetime.now().strftime("%Y%b%d-%H%M%S")
        if logging_params['logging_dir'] is None:
            self.ckpt_path = os.path.join(logging_params['ckpt_path'], run_timestamp)
        else:
            self.ckpt_path = os.path.join(logging_params['ckpt_path'], logging_params['logging_dir'])
        if logging_params['use_logging']:
            from utils.log_util import get_logger
            from utils.parse_util import format_config

            if not os.path.exists(self.ckpt_path):
                os.makedirs(self.ckpt_path)
            self.logger = get_logger(os.path.join(self.ckpt_path, '{}.log'.format(network_params['net_name'])))
            self.logger.info(">>>The config is:")
            self.logger.info(format_config(config))
            self.logger.info(">>>The net is:")
            self.logger.info(self.net)
        if logging_params['use_tensorboard']:
            from torch.utils.tensorboard import SummaryWriter

            if logging_params['logging_dir'] is None:
                self.run_path = os.path.join(logging_params['run_path'], run_timestamp)
            else:
                self.run_path = os.path.join(logging_params['run_path'], logging_params['logging_dir'])
            if not os.path.exists(self.run_path):
                os.makedirs(self.run_path)
            self.writer = SummaryWriter(self.run_path)

    def run(self):
        optim_params = self.config['optim']
        if optim_params['optim_method'] == 'sgd':
            sgd_params = optim_params['sgd']
            optimizer = optim.SGD(self.net.parameters(),
                                  lr=sgd_params['base_lr'],
                                  momentum=sgd_params['momentum'],
                                  weight_decay=sgd_params['weight_decay'],
                                  nesterov=sgd_params['nesterov'])
        elif optim_params['optim_method'] == 'adam':
            adam_params = optim_params['adam']
            optimizer = optim.Adam(self.net.parameters(),
                                   lr=adam_params['base_lr'],
                                   betas=adam_params['betas'],
                                   weight_decay=adam_params['weight_decay'],
                                   amsgrad=adam_params['amsgrad'])
        elif optim_params['optim_method'] == 'adamW':
            adamW_params = optim_params['adamW']
            optimizer = optim.AdamW(self.net.parameters(),
                                    lr=adamW_params['base_lr'],
                                    betas=adamW_params['betas'],
                                    weight_decay=adamW_params['weight_decay'],
                                    amsgrad=adamW_params['amsgrad'])
        else:
            raise Exception('Not support optim method: {}.'.format(optim_params['optim_method']))

        # choosing whether to use lr_decay and related parameters
        lr_decay = None
        if optim_params['use_lr_decay']:
            from torch.optim import lr_scheduler
            if optim_params['lr_decay_method'] == 'cosine':
                cosine_params = optim_params['cosine']
                lr_decay = lr_scheduler.CosineAnnealingLR(
                    optimizer, eta_min=cosine_params['eta_min'], T_max=cosine_params['T_max'])
            elif optim_params['lr_decay_method'] == 'exponent':
                exponent_params = optim_params['exponent']
                lr_decay = lr_scheduler.ExponentialLR(
                    optimizer, gamma=exponent_params['gamma'])
            elif optim_params['lr_decay_method'] == 'warmup':
                warmup_params = optim_params['warmup']
                from nets.cores.warmup_scheduler import GradualWarmupScheduler
                if warmup_params['after_scheduler'] == 'cosine':
                    cosine_params = optim_params['cosine']
                    after_scheduler = lr_scheduler.CosineAnnealingLR(
                        optimizer, eta_min=cosine_params['eta_min'], T_max=cosine_params['T_max'])
                elif warmup_params['after_scheduler'] == 'exponent':
                    exponent_params = optim_params['exponent']
                    after_scheduler = lr_scheduler.ExponentialLR(
                        optimizer, gamma=exponent_params['gamma'])
                else:
                    raise Exception('Not support after_scheduler method: {}.'.format(warmup_params['after_scheduler']))

                lr_decay = GradualWarmupScheduler(optimizer, multiplier=warmup_params['multiplier'],
                                                  total_epoch=warmup_params['total_epoch'],
                                                  after_scheduler=after_scheduler)

        data_params = self.config['data']

        # making train dataset and dataloader
        train_params = self.config['train']
        train_trans_seq = self._resolve_transforms(train_params['aug_trans'])
        train_dataset = CRCDataset3DAE(
            data_root=data_params['data_root'],
            sample_csv=data_params['train_record_csv'],
            transforms=train_trans_seq, )
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_params['batch_size'],
            shuffle=True,
            num_workers=train_params['num_workers'],
            pin_memory=train_params['pin_memory'],
            drop_last=True)

        # making eval dataset and dataloader
        eval_params = self.config['eval']
        eval_trans_seq = self._resolve_transforms(eval_params['aug_trans'])
        eval_dataset = CRCDataset3DAE(
            data_root=data_params['data_root'],
            sample_csv=data_params['eval_record_csv'],
            transforms=eval_trans_seq)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=eval_params['batch_size'],
            shuffle=False,
            num_workers=eval_params['num_workers'],
            pin_memory=eval_params['pin_memory'],
            drop_last=True)

        # choosing criterion
        criterion_params = self.config['criterion']
        if criterion_params['criterion_method'] == 'mse_loss':
            criterion = nn.MSELoss().to(self.device)
        elif criterion_params['criterion_method'] == 'ae_loss':
            criterion = AELoss().to(self.device)
        else:
            raise Exception('Not support criterion method: {}.'
                            .format(criterion_params['criterion_method']))

        # recording the best model
        best_metric = 100
        best_epoch = 0
        for epoch_id in range(optim_params['num_epochs']):
            self._train(epoch_id, train_loader, criterion, optimizer)
            if optim_params['use_lr_decay']:
                lr_decay.step()
                for param_group in optimizer.param_groups:
                    self.logger.info("[Info] epoch:{}, lr {:.6f}".format(epoch_id, param_group['lr']))
            eval_metric = self._eval(epoch_id, eval_loader, criterion)

            # saving the best model
            if eval_metric <= best_metric:
                best_metric = eval_metric
                best_epoch = epoch_id
                self.save(epoch_id)
            self.logger.info('[Info] The maximal metric is {:.4f} at epoch {}'.format(
                best_metric,
                best_epoch))

    def _train(self, epoch_id, data_loader, criterion, optimizer):
        loss_meter = AverageMeter()

        self.net.train()
        with tqdm(total=len(data_loader)) as pbar:
            for batch_id, sample in enumerate(data_loader):
                image = sample['image'].to(self.device)
                optimizer.zero_grad()
                logit = self.net(image)
                loss = criterion(logit, image)
                loss.backward()
                optimizer.step()
                loss_meter.update(loss.data.item(), image.size(0))

                pbar.update(1)
                pbar.set_description("[Train] Epoch:{}, Loss:{:.4f}".format(epoch_id, loss_meter.avg))

        logging_params = self.config['logging']
        if logging_params['use_logging']:
            self.logger.info("[Train] Epoch:{}, Loss:{:.4f}".format(epoch_id, loss_meter.avg))
        if logging_params['use_tensorboard']:
            self.writer.add_scalar('train/loss', loss_meter.avg, epoch_id)

    def _eval(self, epoch_id, data_loader, criterion):
        loss_meter = AverageMeter()
        self.net.eval()
        with torch.no_grad():
            with tqdm(total=len(data_loader)) as pbar:
                for batch_id, sample in enumerate(data_loader):
                    image = sample['image'].to(self.device)
                    logit = self.net(image)
                    loss = criterion(logit, image)
                    loss_meter.update(loss.data.item(), image.size(0))
                    pbar.update(1)
                    pbar.set_description("[Eval] Epoch:{}, Loss:{:.4f}".format(epoch_id, loss_meter.avg))

            logging_params = self.config['logging']
            if logging_params['use_logging']:
                self.logger.info("[Eval] Epoch:{}, Loss:{:.4f}".format(epoch_id, loss_meter.avg))
            if logging_params['use_tensorboard']:
                self.writer.add_scalar('train/loss', loss_meter.avg, epoch_id)

        return loss_meter.avg

    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=torch.device(self.config['network']['device']))
        if self.config['network']['use_parallel']:
            self.net.module.load_state_dict(ckpt)
        else:
            self.net.load_state_dict(ckpt)
        print(">>> Loading model successfully from {}.".format(ckpt_path))

    def save(self, epoch):
        if self.config['network']['use_parallel']:
            state_dict = self.net.module.state_dict()
        else:
            state_dict = self.net.state_dict()
        torch.save(state_dict, os.path.join(self.ckpt_path, '{}.pth'.format(epoch)))

    def _resolve_transforms(self, aug_trans_params):
        """
            According to the given parameters, resolving transform methods
        :param aug_trans_params: the json of transform methods used
        :return: the list of augment transform methods
        """
        trans_seq = []
        for trans_name in aug_trans_params['trans_seq']:
            if trans_name == 'to_tensor':
                trans_seq.append(CRCT3d.ToTensor())
            elif trans_name == 'resize':
                params = aug_trans_params['resize']
                trans_seq.append(CRCT3d.Resize(params['size']))
            elif trans_name == 'normalize':
                params = aug_trans_params['normalize']
                trans_seq.append(CRCT3d.Normalize(params['mean'], params['std']))
            else:
                raise Exception('Not support transform method: {}.'.format(trans_name))

        return Compose(trans_seq)
