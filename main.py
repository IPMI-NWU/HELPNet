import os
import argparse
import datetime
import random
import time
from pathlib import Path
from tensorboardX import SummaryWriter
import numpy as np
import torch
from engines import train_one_epoch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
import util.misc as utils
from data import build
from inference import infer
from models import build_model
from torch.autograd import Variable


def convert_targets(targets):
    masks = [t["masks"] for t in targets]
    target_masks = torch.stack(masks)
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 4, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks


@torch.no_grad()
def evaluate(model, criterion, dataloader_dict, device, visualizer, epoch, writer):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 20
    numbers = {k: len(v) for k, v in dataloader_dict.items()}
    dataloader = dataloader_dict['MR']
    tasks = dataloader_dict.keys()
    counts = {k: 0 for k in tasks}
    total_steps = sum(numbers.values())
    start_time = time.time()
    sample_list, output_list, target_list = [], [], []
    step = 0
    for samples, targets in dataloader:
        start = time.time()
        tasks = [t for t in tasks if counts[t] < numbers[t]]
        task = random.sample(tasks, 1)[0]
        counts.update({task: counts[task] + 1})
        datatime = time.time() - start
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in targets]
        targets_onehot = convert_targets(targets)
        samples_var = Variable(samples.tensors, requires_grad=True)
        outputs = model(samples_var, task)
        loss_dict = criterion(outputs, targets_onehot)
        weight_dict = criterion.weight_dict
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        metric_logger.update(loss=loss_dict_reduced_scaled['loss_CrossEntropy'], **loss_dict_reduced_scaled)
        itertime = time.time() - start
        metric_logger.log_every(step, total_steps, datatime, itertime, print_freq, header)
        if step == 0:
            sample_list.append(samples_var[0])
            _, pre_masks = torch.max(outputs['pred_masks'][0], 0, keepdims=True)
            output_list.append(pre_masks)
            target_list.append(targets_onehot.argmax(1, keepdim=True)[0])
        step = step + 1
    # gather the stats from all processes
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / total_steps))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    writer.add_scalar('avg_DSC', stats['Avg'], epoch)
    visualizer(torch.stack(sample_list), torch.stack(output_list), torch.stack(target_list), epoch, writer)
    return stats


def get_args_parser():
    tasks = {'MR': {'lab_values': [0, 1, 2, 3, 4, 5], 'out_channels': 4}}
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--lr_drop', default=500, type=int)
    parser.add_argument('--tasks', default=tasks, type=dict)
    parser.add_argument('--model', default='MSCMR', required=False)

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None, help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--in_channels', default=1, type=int)

    # * Loss coefficients
    parser.add_argument('--multiDice_loss_coef', default=0, type=float)
    parser.add_argument('--CrossEntropy_loss_coef', default=1, type=float)
    parser.add_argument('--Rv', default=1, type=float)
    parser.add_argument('--Lv', default=1, type=float)
    parser.add_argument('--Myo', default=1, type=float)
    parser.add_argument('--Avg', default=1, type=float)

    # dataset parameters
    parser.add_argument('--dataset', default='MSCMR_dataset', type=str, help='dataset')
    # set your outputdir 
    parser.add_argument('--output_dir', default='output/test/', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', type=str, help='device to use for training / testing')
    parser.add_argument('--GPU_ids', type=str, default='0', help='Ids of GPUs')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):
    if not args.eval:
        writer = SummaryWriter(log_dir=args.output_dir + '/summary')
    args.mean = torch.tensor([0.5], dtype=torch.float32).reshape(1, 1, 1, 1).cuda()
    args.std = torch.tensor([0.5], dtype=torch.float32).reshape(1, 1, 1, 1).cuda()
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    losses = ['CrossEntropy']
    model, criterion, postprocessors, visualizer = build_model(args, losses)
    model.to(device)

    if args.eval:
        infer(model, device)
        return

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [{"params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad]}]
    optimizer = torch.optim.Adam(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    print('Building training dataset...')
    dataset_train_dict = build(image_set='train', args=args)
    num_train = [len(v) for v in dataset_train_dict.values()]
    print('Number of training images: {}'.format(sum(num_train)))

    print('Building validation dataset...')
    dataset_val_dict = build(image_set='val', args=args)
    num_val = [len(v) for v in dataset_val_dict.values()]
    print('Number of validation images: {}'.format(sum(num_val)))

    sampler_train_dict = {k: torch.utils.data.RandomSampler(v) for k, v in dataset_train_dict.items()}
    sampler_val_dict = {k: torch.utils.data.SequentialSampler(v) for k, v in dataset_val_dict.items()}

    batch_sampler_train = {
        k: torch.utils.data.BatchSampler(v, args.batch_size, drop_last=True) for k, v in sampler_train_dict.items()
    }
    dataloader_train_dict = {
        k: DataLoader(v1, batch_sampler=v2, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        for (k, v1), v2 in zip(dataset_train_dict.items(), batch_sampler_train.values())
    }
    dataloader_val_dict = {
        k: DataLoader(v1, args.batch_size, sampler=v2, drop_last=False, collate_fn=utils.collate_fn,
                      num_workers=args.num_workers)
        for (k, v1), v2 in zip(dataset_val_dict.items(), sampler_val_dict.values())
    }

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.whst.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)

    print("Start training")
    start_time = time.time()
    best_dice = None
    for epoch in range(args.epochs):
        print('-' * 40)
        train_stats = train_one_epoch(model, criterion, dataloader_train_dict, optimizer, device, epoch, args, writer)

        lr_scheduler.step()

        # evaluate
        losses = ['CrossEntropy', 'Rv', 'Lv', 'Myo', 'Avg']
        _, criterion, _, _ = build_model(args, losses)

        test_stats = evaluate(model, criterion, dataloader_val_dict, device, visualizer, epoch, writer)
        # save checkpoint for high dice score
        dice_score = test_stats["Avg"]
        print("dice score:", dice_score)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if best_dice is None or dice_score > best_dice:
                best_dice = dice_score
                print("Update best model!")
                checkpoint_paths.append(output_dir / 'best_checkpoint.pth')

            # You can change the threshold
            if dice_score > 0.70:
                print("Update high dice score model!")
                file_name = str(dice_score)[0:6] + '_' + str(epoch) + '_checkpoint.pth'
                checkpoint_paths.append(output_dir / file_name)

            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'args': args,
                }, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MSCMR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.GPU_ids)
    main(args)
