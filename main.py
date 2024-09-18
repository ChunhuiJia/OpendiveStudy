import os
import sys
import time
import random
from tqdm import tqdm   # 进度条
from argparse import ArgumentParser  # 参数解析器

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

if torch.__version__ == 'parrots':
    from pavi import SummaryWriter
else:
    from torch.utils.tensorboard import SummaryWriter

from data import PlanningDataset, SequencePlanningDataset, Comma2k19SequenceDataset
from model import PlaningNetwork, MultipleTrajectoryPredictionLoss, SequencePlanningNetwork
from utils import draw_trajectory_on_ax, get_val_metric, get_val_metric_keys
os.environ['PORT'] = '23333'
os.environ['SLURM_PROCID'] = '0'
os.environ['SLURM_NTASKS'] = '1'


def get_hyperparameters(parser: ArgumentParser):
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log_per_n_step', type=int, default=20)
    parser.add_argument('--val_per_n_epoch', type=int, default=1)

    parser.add_argument('--resume', type=str, default='')

    parser.add_argument('--M', type=int, default=5)
    parser.add_argument('--num_pts', type=int, default=33)
    parser.add_argument('--mtp_alpha', type=float, default=1.0)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--sync_bn', type=bool, default=True)
    parser.add_argument('--tqdm', type=bool, default=True)
    parser.add_argument('--optimize_per_n_step', type=int, default=40)

    try:
        exp_name = os.environ["SLURM_JOB_ID"]
    except KeyError:
        exp_name = str(time.time())
    parser.add_argument('--exp_name', type=str, default=exp_name)

    return parser


def setup(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method='tcp://localhost:%s' % os.environ['PORT'], rank=rank, world_size=world_size)
    print('[%.2f]' % time.time(), 'DDP Initialized at %s:%s' % ('localhost', os.environ['PORT']), rank, 'of', world_size, flush=True)


def get_dataloader(rank, world_size, batch_size, pin_memory=False, num_workers=0):
    train = Comma2k19SequenceDataset('data/comma2k19_train_non_overlap.txt', 'data/comma2k19/','train', use_memcache=False)
    val = Comma2k19SequenceDataset('data/comma2k19_val_non_overlap.txt', 'data/comma2k19/','demo', use_memcache=False)

    if torch.__version__ == 'parrots':
        dist_sampler_params = dict(num_replicas=world_size, rank=rank, shuffle=True)
    else:
        dist_sampler_params = dict(num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_sampler = DistributedSampler(train, **dist_sampler_params)
    val_sampler = DistributedSampler(val, **dist_sampler_params)

    loader_args = dict(num_workers=num_workers, persistent_workers=True if num_workers > 0 else False, prefetch_factor=2, pin_memory=pin_memory)
    train_loader = DataLoader(train, batch_size, sampler=train_sampler, **loader_args)
    val_loader = DataLoader(val, batch_size=1, sampler=val_sampler, **loader_args)

    return train_loader, val_loader


def cleanup():
    dist.destroy_process_group()

class SequenceBaselineV1(nn.Module):
    def __init__(self, M, num_pts, mtp_alpha, lr, optimizer, optimize_per_n_step=40) -> None:
        super().__init__()
        self.M = M
        self.num_pts = num_pts
        self.mtp_alpha = mtp_alpha
        self.lr = lr
        self.optimizer = optimizer

        self.net = SequencePlanningNetwork(M, num_pts)

        self.optimize_per_n_step = optimize_per_n_step  # for the gru module

    @staticmethod
    def configure_optimizers(args, model):
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
        elif args.optimizer == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, )
        else:
            raise NotImplementedError
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 20, 0.9)

        return optimizer, lr_scheduler

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros((2, x.size(0), 512)).to(self.device)
        return self.net(x, hidden)


def main(rank, world_size, args):
    if rank == 0:
        writer = SummaryWriter()

    train_dataloader, val_dataloader = get_dataloader(rank, world_size, args.batch_size, False, args.n_workers)
    model = SequenceBaselineV1(args.M, args.num_pts, args.mtp_alpha, args.lr, args.optimizer, args.optimize_per_n_step)
    use_sync_bn = args.sync_bn
    if use_sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()
    optimizer, lr_scheduler = model.configure_optimizers(args, model)
    model: SequenceBaselineV1
    if args.resume and rank == 0:
        print('Loading weights from', args.resume)
        model.load_state_dict(torch.load(args.resume), strict=True)
    dist.barrier()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)
    loss = MultipleTrajectoryPredictionLoss(args.mtp_alpha, args.M, args.num_pts, distance_type='angle')

    num_steps = 0
    disable_tqdm = (not args.tqdm) or (rank != 0)

    for epoch in tqdm(range(args.epochs), disable=disable_tqdm, position=0):
        train_dataloader.sampler.set_epoch(epoch)
        
        for batch_idx, data in enumerate(tqdm(train_dataloader, leave=False, disable=disable_tqdm, position=1)):
            seq_inputs, seq_labels = data['seq_input_img'].cuda(), data['seq_future_poses'].cuda()
            bs = seq_labels.size(0)
            seq_length = seq_labels.size(1)
            
            hidden = torch.zeros((2, bs, 512)).cuda()   # 为什么要单独定义一个隐藏层呢？？？或者说这里的作用是把隐藏层置0？？？GRU的隐藏层需要自己定义？？？
            total_loss = 0
            for t in tqdm(range(seq_length), leave=False, disable=disable_tqdm, position=2):
                num_steps += 1
                inputs, labels = seq_inputs[:, t, :, :, :], seq_labels[:, t, :, :]
                # 输入input和hidden状态进行前向推理的计算
                pred_cls, pred_trajectory, hidden = model(inputs, hidden)  # 有可能隐藏状态比较特殊，是包含了一些历史信息，所以需要对隐藏层保留或初始化，在SequenceBaselineV1的forward(self, x, hidden=None)前向传播函数中定义了hidden的输入

                cls_loss, reg_loss = loss(pred_cls, pred_trajectory, labels)  # 计算损失，但是没看懂交叉熵损失cls_loss,没看出来是分类的问题啊。。。
                total_loss += (cls_loss + args.mtp_alpha * reg_loss.mean()) / model.module.optimize_per_n_step  # n个batch批次的损失的平均值作为总的损失来计算梯度来更新模型参数
            
                if rank == 0 and (num_steps + 1) % args.log_per_n_step == 0:
                    # TODO: add a customized log function
                    writer.add_scalar('train/epoch', epoch, num_steps)
                    writer.add_scalar('loss/cls', cls_loss, num_steps)
                    writer.add_scalar('loss/reg', reg_loss.mean(), num_steps)
                    writer.add_scalar('loss/reg_x', reg_loss[0], num_steps)
                    writer.add_scalar('loss/reg_y', reg_loss[1], num_steps)
                    writer.add_scalar('loss/reg_z', reg_loss[2], num_steps)
                    writer.add_scalar('param/lr', optimizer.param_groups[0]['lr'], num_steps)
                
                # 经过一次（多个批次作为一次）loss的计算后将进行一次反向传播的参数迭代，并将cost重置为0
                if (t + 1) % model.module.optimize_per_n_step == 0:
                    hidden = hidden.clone().detach()  # hidden.clone()：复制 hidden 张量中的数据，并返回一个新的张量，.detach()表示其从当前计算图中分离，不参与到反向传播中
                    optimizer.zero_grad() # 是一个非常重要的步骤，它用于在每次训练迭代开始时清除（归零）模型参数的梯度。这是训练神经网络时的一个标准操作，因为默认情况下，PyTorch 会累积梯度，而不是在每次迭代后立即清除它们。
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # TODO: move to args ，是一个用于梯度裁剪的函数，它可以帮助防止梯度爆炸问题
                    optimizer.step()
                    if rank == 0:
                        writer.add_scalar('loss/total', total_loss, num_steps)
                    total_loss = 0

            if not isinstance(total_loss, int): # isinstance() 函数用于检查一个对象是否是一个已知的类型。这个函数通常用来做类型检查，以确保你正在处理正确类型的数据
                # 为什么要检查 total_loss 是不是 int 类型？？？
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # TODO: move to args
                optimizer.step()
                if rank == 0:
                    writer.add_scalar('loss/total', total_loss, num_steps)

        # 一个 epoch 之后，通过 lr_scheduler 来更新歩长 lr
        lr_scheduler.step()
        if (epoch + 1) % args.val_per_n_epoch == 0:   # n 个 epoch 之后进行一次模型的验证测试
            if rank == 0:
                # save model
                ckpt_path = os.path.join(writer.log_dir, 'epoch_%d.pth' % epoch)
                torch.save(model.module.state_dict(), ckpt_path)  # 需要保存模型的状态字典（即模型的参数）时，你应该使用 model.module.state_dict() 来获取它
                print('[Epoch %d] checkpoint saved at %s' % (epoch, ckpt_path))

            # 在 PyTorch 中，model.eval() 是一个非常重要的方法，用于将模型设置为评估模式（evaluation mode）。这通常在模型的推理阶段或验证阶段使用，以确保模型的行为与训练阶段有所不同。
            # 主要作用：
            # <禁用 Dropout>：在训练期间，Dropout 层会随机丢弃一部分神经元，以防止过拟合。而在评估模式下，Dropout 会被禁用，所有神经元都会被使用。
            # <使用固定的 Batch Normalization 统计量>：在训练期间，Batch Normalization 层会根据当前批次的均值和方差来标准化数据。而在评估模式下，它会使用在训练期间计算的运行均值和方差。
            model.eval()
            
            # 在 PyTorch 中，with torch.no_grad(): 是一个上下文管理器，用于临时禁用在代码块内部的所有计算图和梯度计算。这通常用于模型的评估阶段或进行推理时，因为这些时候<不需要进行反向传播，因此不需要计算梯度>。
            # 主要优点:
            # <节省内存>：梯度计算通常需要额外的内存来存储梯度值。禁用梯度计算可以减少内存消耗。
            # <提高速度>：不计算梯度可以减少计算量，从而加快模型的前向传播速度。
            with torch.no_grad():
                saved_metric_epoch = get_val_metric_keys()
                for batch_idx, data in enumerate(tqdm(val_dataloader, leave=False, disable=disable_tqdm, position=1)):
                    seq_inputs, seq_labels = data['seq_input_img'].cuda(), data['seq_future_poses'].cuda()

                    bs = seq_labels.size(0)
                    seq_length = seq_labels.size(1)
                    
                    hidden = torch.zeros((2, bs, 512), device=seq_inputs.device) # device=seq_inputs.device：指定新创建的张量应该位于与 seq_inputs 相同的设备上。seq_inputs.device 返回 seq_inputs 张量所在的设备，这可以是 CPU 或 GPU。这样做是为了确保张量与模型输入数据在同一设备上，从而避免不必要的设备间数据传输
                    for t in tqdm(range(seq_length), leave=False, disable=True, position=2):
                        inputs, labels = seq_inputs[:, t, :, :, :], seq_labels[:, t, :, :]
                        pred_cls, pred_trajectory, hidden = model(inputs, hidden)
                        # .view() 方法用于改变张量的形状（shape），而不改变其数据
                        metrics = get_val_metric(pred_cls, pred_trajectory.view(-1, args.M, args.num_pts, 3), labels)  # metrics 是一个得分的字典，里面存储了很多得分，方便后面展示
                        
                        for k, v in metrics.items():
                            saved_metric_epoch[k].append(v.float().mean().item())
                
                dist.barrier()  # Wait for all processes
                # sync
                metric_single = torch.zeros((len(saved_metric_epoch), ), dtype=torch.float32, device='cuda')  # 逗号后面没有值，则默认值就是1
                counter_single = torch.zeros((len(saved_metric_epoch), ), dtype=torch.int32, device='cuda')
                # From Python 3.6 onwards, the standard dict type maintains insertion order by default.
                # But, programmers should not rely on it.
                for i, k in enumerate(sorted(saved_metric_epoch.keys())):  # sorted() 函数则将这些键进行排序
                    metric_single[i] = np.mean(saved_metric_epoch[k])
                    counter_single[i] = len(saved_metric_epoch[k])

                metric_gather = [torch.zeros((len(saved_metric_epoch), ), dtype=torch.float32, device='cuda')[None] for _ in range(world_size)]  # world_size，这通常指的是在分布式训练中参与计算的进程数。
                counter_gather = [torch.zeros((len(saved_metric_epoch), ), dtype=torch.int32, device='cuda')[None] for _ in range(world_size)]
                dist.all_gather(metric_gather, metric_single[None])  # 在 PyTorch 的分布式训练环境中，dist.all_gather 是一个集合通信操作，它用于收集所有进程中的张量数据，并将它们合并到每个进程的张量中。这个操作确保了所有进程在执行完 all_gather 后，都有了完整的数据集。
                dist.all_gather(counter_gather, counter_single[None])

                if rank == 0:
                    metric_gather = torch.cat(metric_gather, dim=0)  # [world_size, num_metric_keys]  # 将收集到的数据合并成一个张量
                    counter_gather = torch.cat(counter_gather, dim=0)  # [world_size, num_metric_keys]
                    metric_gather_weighted_mean = (metric_gather * counter_gather).sum(0) / counter_gather.sum(0)
                    for i, k in enumerate(sorted(saved_metric_epoch.keys())):
                        writer.add_scalar(k, metric_gather_weighted_mean[i], num_steps)
                dist.barrier()

            model.train()

    cleanup()


if __name__ == "__main__":
    print('[%.2f]' % time.time(), 'starting job...', os.environ['SLURM_PROCID'], 'of', os.environ['SLURM_NTASKS'], flush=True)

    parser = ArgumentParser()
    parser = get_hyperparameters(parser)
    args = parser.parse_args(args=[])

    setup(rank=int(os.environ['SLURM_PROCID']), world_size=int(os.environ['SLURM_NTASKS']))
    main(rank=int(os.environ['SLURM_PROCID']), world_size=int(os.environ['SLURM_NTASKS']), args=args)
