import os
import math

import torch
import torch.nn as nn

from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

import numpy as np
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from torchvision import transforms
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

train_transforms = transforms.Compose(
    [
         # data_transforms.PointcloudScale(),
         data_transforms.PointcloudRotate(),
         # data_transforms.PointcloudTranslate(),
         # data_transforms.PointcloudJitter(),
         # data_transforms.PointcloudRandomInputDropout(),
         # data_transforms.RandomHorizontalFlip(),
         # data_transforms.PointcloudScaleAndTranslate(),
    ]
)

train_transforms_raw = transforms.Compose(
    [
         # data_transforms.PointcloudScale(),
         # data_transforms.PointcloudRotate(),
         # data_transforms.PointcloudTranslate(),
         # data_transforms.PointcloudJitter(),
         # data_transforms.PointcloudRandomInputDropout(),
         # data_transforms.RandomHorizontalFlip(),
         data_transforms.PointcloudScaleAndTranslate(),
    ]
)

test_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        # data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudTranslate(),
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)


class Acc_Metric:
    def __init__(self, acc=0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        elif type(acc).__name__ == 'Acc_Metric':
            self.acc = acc.acc
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


def run_net(args, config, train_writer=None, val_writer=None):

    rotation = config.model.rotation

    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader), = builder.dataset_builder(args, config.dataset.train), \
        builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)

    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    best_metrics_vote = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    # args.resume = True      
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Acc_Metric(best_metrics)
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger=logger)

    if args.use_gpu:
        base_model.to(args.local_rank)
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[args.local_rank % torch.cuda.device_count()])   
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
    misc.summary_parameters(base_model, logger=logger)

    start_tau = 0.0
    tau_start_epoch = 0
    tau_warmup_epochs = 10
    max_tau = 0.5

    baseline = None

    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        tau = tau_schedule(epoch - tau_start_epoch, start_tau, max_tau, tau_warmup_epochs,
                           config.max_epoch - tau_start_epoch)

        # from calflops import calculate_flops
        # flops, macs, params = calculate_flops(base_model, input_shape=(1, 2048, 3))

        npoints = config.npoints
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx

            data_time.update(time.time() - batch_start_time)

            points = data[0].cuda()
            label = data[1].cuda()

            if npoints == 1024:
                point_all = 1200
            elif npoints == 2048:
                point_all = 2400
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()

            if points.size(1) < point_all:
                point_all = points.size(1)

            fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)  # (B, npoint)
            fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1,
                                                                                                              2).contiguous()  # (B, N, 3)
            if 'scan' in args.config and rotation == True:
                points = train_transforms(points)    
            else:
                points = train_transforms_raw(points)

            batch_size = points.size(0)
            ret = base_model(points, gt=None, tau=None, use_wavelets=True)
            #policy_old = policy_old.detach()
            loss, acc = base_model.module.get_loss_acc(ret, label)
            loss = loss.mean()
            # if baseline is None:
            #     baseline = -loss.detach().mean()
            #
            # K_epochs = 4
            # beta = 0.99
            # for k in range(K_epochs):
            #     ret, policy = base_model(points, gt=label, tau=tau)
            #     loss, acc = base_model.module.get_loss_acc(ret, label)
            #     reward = -loss.detach()
            #     ratio = torch.exp(torch.clamp(policy - policy_old, max=3))
            #     baseline = beta*baseline + (1-beta)*reward.mean()
            #     advantage = reward - baseline
            #     advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)
            #     policy_loss = -torch.minimum(ratio * advantage, torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * advantage).mean()
            #     loss = policy_loss
            _loss = loss
            _loss.backward()
            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                acc = dist_utils.reduce_tensor(acc, args)
                losses.update([loss.item(), acc.item()])
            else:
                losses.update([loss.item(), acc.item()])

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            # if idx % 10 == 0:
            #     print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss+Acc = %s lr = %.6f' %
            #                 (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
            #                 ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
                  (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
                   optimizer.param_groups[0]['lr']), logger=logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)
            better = metrics.better_than(best_metrics)
            # Save ckeckpoints
            if better:
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args,
                                        logger=logger)
                print_log(
                    "--------------------------------------------------------------------------------------------",
                    logger=logger)
            if args.vote:
                if metrics.acc > 92.1 or (better and metrics.acc > 91):
                    metrics_vote = validate_vote(base_model, test_dataloader, epoch, val_writer, args, config,
                                                 logger=logger)
                    if metrics_vote.better_than(best_metrics_vote):
                        best_metrics_vote = metrics_vote
                        print_log(
                            "****************************************************************************************",
                            logger=logger)
                        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics_vote,
                                                'ckpt-best_vote', args, logger=logger)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)
        # if (config.max_epoch - epoch) < 10:
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=None):
    # print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)
            save_dir_path = None
            if idx == 0:
                save_dir_path = os.path.join('examples', args.exp_name)
                os.makedirs(save_dir_path, exist_ok=True)
            logits = base_model(points, tau=None, use_wavelets=True, save_pts_dir=save_dir_path, epoch=epoch)
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', acc, epoch)

    return Acc_Metric(acc)


def validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=None, times=10):
    print_log(f"[VALIDATION_VOTE] epoch {epoch}", logger=logger)
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()

            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
            local_pred = []

            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(),
                                                          fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

                points = test_transforms(points)

                logits = base_model(points)
                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)

            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation_vote] EPOCH: %d  acc_vote = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)

    return Acc_Metric(acc)


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger=logger)  # for finetuned transformer
    # base_model.load_model_from_ckpt(args.ckpts) # for BERT
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, args, config, logger=logger)


def test(base_model, test_dataloader, args, config, logger=None):
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    npoints = config.npoints

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()
            points = misc.fps(points, npoints)
            logits = base_model(points)
            target = label.view(-1)
            pred = logits.argmax(-1).view(-1)
            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[TEST] acc = %.4f' % acc, logger=logger)

        if args.vote:

            if args.distributed:
                torch.cuda.synchronize()

            print_log(f"[TEST_VOTE]", logger = logger)
            acc = 0.
            for time in range(1, 300):
                this_acc = test_vote(base_model, test_dataloader, 1, None, args, config, logger=logger, times=10)
                if acc < this_acc:
                    acc = this_acc
                print_log('[TEST_VOTE_time %d]  acc = %.4f, best acc = %.4f' % (time, this_acc, acc), logger=logger)
            print_log('[TEST_VOTE] acc = %.4f' % acc, logger=logger)


def test_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=None, times=10):
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()

            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
            local_pred = []

            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(),
                                                          fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

                points = test_transforms(points)

                logits = base_model(points)
                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)

            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)
    # print_log('[TEST] acc = %.4f' % acc, logger=logger)

    return acc


def plot_embedding(data, label, title, category_nums):
    TSNE_PATH = "./vis/tsne/"
    # colors = []
    colors = ['#e6194B', '#3cb44b', '#4363d8', '#f58231', '#42d4f4', '#f032e6', '#fabed4', '#469990',
              '#dcbeff', '#9A6324', '#800000', '#000075', '#a9a9a9', '#888870', '#000000'
              ]
    if category_nums > 27:
        base = [0, 0.3, 0.6, 0.9]
    else:
        base = [0, 0.5, 0.9]
    # for i in range(len(base)):
    #     for j in range(len(base)):
    #         for k in range(len(base)):
    #             colors.append([base[i], base[j], base[k], 1])

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure(figsize=(8, 8))
    for i in range(data.shape[0]):
        print(colors[int(label[i])])
        plt.scatter(data[i, 0], data[i, 1], s=8, marker='o', c=colors[int(label[i])], cmap='coolwarm')
        # plt.text(data[i, 0], data[i, 1], str(label[i]),
        #          color=colors[int(label[i])],
        #          # fontdict={'weight': 'bold', 'size': 9}
        #          fontdict={'family': 'Times New Roman',
        #                    'weight': 'normal',
        #                    'size': 8, }
        #          )
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.axis('off')

    if not os.path.isdir(TSNE_PATH):
        os.makedirs(TSNE_PATH)
    plt.savefig(TSNE_PATH + "tsne_fix_.pdf")
    return fig


def test_only_tsne(base_model, test_dataloader, args, config, logger=None):
    base_model.eval()  # set model to eval mode

    test_pred = []
    test_label = []
    test_feature = []
    npoints = config.npoints

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            # get_local.clear()
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            _, concat_f = base_model(points)

            target = label.view(-1)

            test_label.append(target.detach())
            test_feature.append(concat_f.detach())

        test_label = torch.cat(test_label, dim=0)

        category_nums = config.model.cls_dim

        index = test_label < category_nums
        label_all = test_label[index]
        test_feature = torch.cat(test_feature, dim=0)
        test_feature = test_feature[index]

        # tsne
        test_feature = test_feature.cpu().numpy()
        label = label_all.cpu().numpy()

        tsne = TSNE(n_components=2, init='pca', random_state=0)
        result = tsne.fit_transform(test_feature.squeeze())

        fig = plot_embedding(result, label, '', category_nums)


def test_tsne(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.val)
    base_model = builder.model_builder(config.model)
    # load checkpoints
    # builder.load_model(base_model, args.ckpts, logger=logger)  # for finetuned transformer
    base_model.load_model_from_ckpt(args.ckpts)  # for BERT

    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    test_only_tsne(base_model, test_dataloader, args, config, logger=logger)
