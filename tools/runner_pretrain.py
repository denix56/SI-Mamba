import math
import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import numpy as np
from torchvision import transforms
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
import itertools
import random

train_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        # data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudRotatePerturbation(),
        # data_transforms.PointcloudTranslate(),
        # data_transforms.PointcloudJitter(),
        # data_transforms.PointcloudRandomInputDropout(),
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)


def tau_schedule(epoch, start_tau, max_tau, warmup_epochs, total_epochs):
    if epoch < warmup_epochs:
        # Linear warmup
        progress = epoch / warmup_epochs
        return start_tau + (max_tau - start_tau) * progress
    elif epoch <= total_epochs:
        # Cosine annealing: from max_tau to 0
        anneal_progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return max_tau * 0.5 * (1 + math.cos(math.pi * anneal_progress))
    else:
        return 0.0


class Acc_Metric:
    def __init__(self, acc=0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def evaluate_svm(train_features, train_labels, test_features, test_labels):
    # clf = LinearSVC()
    # clf.fit(train_features, train_labels)
    # pred = clf.predict(test_features)
    # return np.sum(test_labels == pred) * 1. / pred.shape[0]

    clf = SVC(C=0.01, kernel='linear')
    train_features = train_features.mean(1) + train_features.max(1)
    clf.fit(train_features, train_labels)
    test_features = test_features.mean(1) + test_features.max(1)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]


def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader), = builder.dataset_builder(args, config.dataset.train), \
        builder.dataset_builder(args, config.dataset.val)
    # (_, extra_train_dataloader), (_, extra_test_dataloader),  = builder.dataset_builder(args, config.dataset.extra_train_svm), \
    #                                                             builder.dataset_builder(args, config.dataset.extra_test_svm)

    extra_train_dataloader, extra_test_dataloader = builder.dataset_builder_svm(config.dataset.svm)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()

    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger=logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[args.local_rank % torch.cuda.device_count()],
                                                         find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    # trainval
    # training
    base_model.zero_grad()
    start_tau = 0.01
    tau_start_epoch = 0
    tau_warmup_epochs = 20
    max_tau = 1.0
    beta = 0.99

    baseline = None

    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss'])

        num_iter = 0
        tau = tau_schedule(epoch - tau_start_epoch, start_tau, max_tau, tau_warmup_epochs,
                           config.max_epoch - tau_start_epoch)

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        # batch_iter = itertools.cycle(train_dataloader)
        # BATCHES_PER_ROLLOUT = 8
        # K_epochs = 4
        # eps = 0.2

        # losses_all = []

        # for idx in range(n_batches):
        #     num_iter += 1
        #     n_itr = epoch * n_batches + idx
        #
        #     data_time.update(time.time() - batch_start_time)
        #     # import ipdb; ipdb.set_trace()
        #     # npoints = config.dataset.train._base_.N_POINTS
        #     npoints = config.npoints
        #     dataset_name = config.dataset.train._base_.NAME
        #
        #     points_batch = []
        #     losses_batch = []
        #     policy_batch = []
        #
        #     for b in range(BATCHES_PER_ROLLOUT):
        #         taxonomy_ids, model_ids, data = next(batch_iter)
        #
        #         if dataset_name == 'ShapeNet':
        #             points = data.cuda()
        #         elif dataset_name == 'ModelNet':
        #             points = data[0].cuda()
        #             points = misc.fps(points, npoints)
        #         else:
        #             raise NotImplementedError(f'Train phase do not support {dataset_name}')
        #
        #         assert points.size(1) == npoints
        #         points = train_transforms(points)
        #         loss, policy = base_model(points, tau=  # None
        #         tau if epoch >= tau_start_epoch else None, ret_policy=True
        #                                   )
        #         points_batch.append(points)
        #         losses_batch.append(loss.view(points.shape[0], -1).mean(1).detach())
        #         policy_batch.append(policy.detach())
        #
        #     losses_batch = torch.stack(losses_batch, 0)
        #     #policy_batch = torch.stack(policy_batch, 0)
        #     if baseline is None:
        #         baseline = -losses_batch.mean()
        #
        #     reward = -losses_batch
        #     baseline = beta * baseline + (1 - beta) * reward.mean()
        #     advantage = reward - baseline
        #     advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)
        #
        #     for k in range(K_epochs):
        #         pairs = list(zip(points_batch, policy_batch))
        #         random.shuffle(pairs)
        #         points_batch, policy_batch = zip(*pairs)
        #
        #         for points, policy_old in zip(points_batch, policy_batch):
        #             policy = base_model(points, tau=tau if epoch >= tau_start_epoch else None, ret_policy=True, ret_only_policy=True)
        #             ratio = torch.exp(torch.clamp(policy - policy_old, max=6))
        #             policy_loss = torch.minimum(ratio * advantage, torch.clamp(ratio, min=1-eps, max=1+eps)*advantage)
        #             policy_loss = -policy_loss.mean()
        #             policy_loss.backward()
        #             optimizer.step()
        #             base_model.zero_grad()
        #             losses_all.append(policy_loss.item())
        #     loss = torch.tensor(sum(losses_all) / len(losses_all))



        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx

            data_time.update(time.time() - batch_start_time)
            # import ipdb; ipdb.set_trace()
            # npoints = config.dataset.train._base_.N_POINTS
            npoints = config.npoints
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            elif dataset_name == 'ModelNet':
                points = data[0].cuda()
                points = misc.fps(points, npoints)
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            assert points.size(1) == npoints
            points = train_transforms(points)
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_bf16_supported(False)):
                loss = base_model(points, tau=tau, ret_policy=False, use_wavelets=True)

            try:
                loss.backward()
                # print("Using one GPU")
            except:
                loss = loss.mean()
                loss.backward()
                # print("Using multi GPUs")

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                losses.update([loss.item() * 1000])
            else:
                losses.update([loss.item() * 1000])

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                          (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                           ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger=logger)

            # break    
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
                  (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
                   optimizer.param_groups[0]['lr']), logger=logger)

        if epoch % args.val_freq == 0:
            #     # Validate the current model
            metrics = validate(base_model, extra_train_dataloader, extra_test_dataloader, epoch, val_writer, args,
                               config, logger=logger)
            #
            #     # Save ckeckpoints
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args,
                                        logger=logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)
        if epoch % 25 == 0 and epoch >= 250:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}',
                                    args,
                                    logger=logger)
        # if (config.max_epoch - epoch) < 10:
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger=None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)
    base_model.eval()  # set model to eval mode

    test_features = []
    test_label = []

    train_features = []
    train_label = []
    npoints = config.dataset.train.others.npoints
    with torch.no_grad():
        # for idx, (taxonomy_ids, model_ids, data) in enumerate(extra_train_dataloader):
        for idx, (points, label) in enumerate(extra_train_dataloader):
            # points = data[0].cuda()
            # label = data[1].cuda()

            points = points.cuda()
            label = label.cuda()

            points = misc.fps(points, npoints)

            assert points.size(1) == npoints
            if idx == 0:
                save_dir_path = os.path.join('examples', args.exp_name)
                os.makedirs(save_dir_path, exist_ok=True)
            feature = base_model(points, noaug=True, tau=None,
                                 use_wavelets=True, save_pts_dir=save_dir_path, epoch=epoch)
            target = label.view(-1)

            train_features.append(feature.detach().cpu())
            train_label.append(target.detach().cpu())

        # for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
        for idx, (points, label) in enumerate(extra_train_dataloader):
            # points = data[0].cuda()
            # label = data[1].cuda()

            points = points.cuda()
            label = label.cuda()

            points = misc.fps(points, npoints)
            assert points.size(1) == npoints
            feature = base_model(points, noaug=True, tau=None,
                                 use_wavelets=True)
            target = label.view(-1)

            test_features.append(feature.detach().cpu())
            test_label.append(target.detach().cpu())

        train_features = torch.cat(train_features, dim=0)
        train_label = torch.cat(train_label, dim=0)
        test_features = torch.cat(test_features, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            train_features = dist_utils.gather_tensor(train_features, args)
            train_label = dist_utils.gather_tensor(train_label, args)
            test_features = dist_utils.gather_tensor(test_features, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        svm_acc = evaluate_svm(train_features.data.cpu().numpy(), train_label.data.cpu().numpy(),
                               test_features.data.cpu().numpy(), test_label.data.cpu().numpy())

        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch, svm_acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', svm_acc, epoch)

    return Acc_Metric(svm_acc)


def test_net():
    pass
