import random
import time
import datetime
import torch.nn.functional as Func
import numpy as np
import torch
import torch.nn as nn
import util.misc as utils
from torch.autograd import Variable
import torchvision
from util.Jigsaw import Jigsaw, RandomBrightnessContrast, Jigsaw_4_4_keep2
from skimage import measure

def keep_largest_connected_components(mask):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''
    heart_slice = np.where((mask > 0), 1, 0)
    out_heart = np.zeros(heart_slice.shape, dtype=np.uint8)
    for struc_id in [1]:
        binary_img = heart_slice == struc_id
        blobs = measure.label(binary_img, connectivity=1)
        props = measure.regionprops(blobs)
        if not props:
            continue
        area = [ele.area for ele in props]
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label
        out_heart[blobs == largest_blob_label] = struc_id

    out_img = np.zeros(mask.shape, dtype=np.uint8)
    for struc_id in [1, 2, 3]:
        binary_img = mask == struc_id
        blobs = measure.label(binary_img, connectivity=1)
        props = measure.regionprops(blobs)
        if not props:
            continue
        area = [ele.area for ele in props]
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label
        out_img[blobs == largest_blob_label] = struc_id

    final_img = out_heart * out_img
    return final_img

class pDLoss(nn.Module):
    def __init__(self, n_classes, ignore_index):
        super(pDLoss, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, ignore_mask):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target * ignore_mask)
        y_sum = torch.sum(target * target * ignore_mask)
        z_sum = torch.sum(score * score * ignore_mask)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None):
        ignore_mask = torch.ones_like(target)
        ignore_mask[target == self.ignore_index] = 0
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i], ignore_mask)
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class Visualize_train(nn.Module):
    def __init__(self):
        super().__init__()

    def save_image(self, image, tag, epoch, writer):
        if tag == 'sample':
            image_max, image_min = 10, -1
            image = (image - image_min) / (image_max - image_min)
            image = torch.clamp(image, 0.0, 1.0)
            grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        else:
            image = (image - image.min()) / (image.max() - image.min() + 1e-6)
            grid = torchvision.utils.make_grid(image, nrow=1, pad_value=1)
        writer.add_image(tag, grid, epoch)

    def forward(self, sample_list, label_list, output_list, epoch, writer):
        self.save_image(sample_list.float(), 'sample', epoch, writer)
        self.save_image(label_list.float(), 'label', epoch, writer)
        self.save_image(output_list.float(), 'output', epoch, writer)


def convert_targets(targets):
    masks = [t["masks"] for t in targets]
    target_masks = torch.stack(masks)
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 5, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks


def to_onehot(target_masks, device):
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 5, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks


def to_onehot_dim4(target_masks, device):
    shp_y = target_masks.shape
    target_masks = target_masks.long()
    y_onehot = torch.zeros((shp_y[0], 4, shp_y[2], shp_y[3]))
    if target_masks.device.type == "cuda":
        y_onehot = y_onehot.cuda(target_masks.device.index)
    y_onehot.scatter_(1, target_masks, 1).float()
    target_masks = y_onehot
    return target_masks


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                       dataloader_dict: dict, optimizer: torch.optim.Optimizer,
                       device: torch.device, epoch: int, args, writer):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    numbers = {k: len(v) for k, v in dataloader_dict.items()}
    dataloader = dataloader_dict['MR']
    tasks = dataloader_dict.keys()
    counts = {k: 0 for k in tasks}
    total_steps = sum(numbers.values())
    start_time = time.time()

    sample_list, label_list, output_list = [], [], []
    step = 0
    for sample, label, edge in dataloader:
        start = time.time()
        tasks = [t for t in tasks if counts[t] < numbers[t]]
        task = random.sample(tasks, 1)[0]
        counts.update({task: counts[task] + 1})
        datatime = time.time() - start

        sample = sample.to(device)
        sample = Variable(sample.tensors, requires_grad=True)
        # -------------------------------------------------------
        # 0:BG, 1:RV, 2:Myo, 3:LV, 4:unlabeled pixels
        # -------------------------------------------------------
        label = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in label]
        label = convert_targets(label)

        edge = [{k: v.to(device) for k, v in t.items() if not isinstance(v, str)} for t in edge]
        edge = torch.stack([t["masks"] for t in edge]) / 255.0

        adv_mask1, adv_mask2 = np.random.binomial(n=1, p=0.1), np.random.binomial(n=1, p=0.1)
        if adv_mask1 == 1 or adv_mask2 == 1:
            noise = torch.zeros_like(sample).uniform_(-1.0 * 10.0 / 255., 10.0 / 255.)
            sample = Variable(sample + noise, requires_grad=True)

        jig_fun = Jigsaw()
        sample_jig_2, shuffle_index_2 = jig_fun(sample, 2, 2)
        jig_fun_4 = Jigsaw_4_4_keep2()
        sample_jig_4, shuffle_index_4 = jig_fun_4(sample_jig_2, 2, 2)

        sample_c = RandomBrightnessContrast(sample, brightness_limit=0.7, contrast_limit=0.7, p=1)

        output_c = model(sample_c, task)
        output_jig_temp_2 = model(sample_jig_2, task)
        output_jig_temp_4 = model(sample_jig_4, task)

        output_jig_2, _ = jig_fun(output_jig_temp_2["pred_masks"], 2, 2, shuffle_index_2)
        output_jig_4s1, _ = jig_fun_4(output_jig_temp_4["pred_masks"], 2, 2, shuffle_index_4)
        output_jig_4, _ = jig_fun(output_jig_4s1, 2, 2, shuffle_index_2)

        output_jig_2 = {'pred_masks': output_jig_2}
        output_jig_4 = {'pred_masks': output_jig_4}

        temperature = 0.4
        softmax = torch.nn.Softmax(dim=1)
        output_weight = softmax(torch.cat([
            torch.sum(output_c["pred_masks"] * torch.log2(output_c["pred_masks"] + 1e-12), dim=1, keepdim=True),
            torch.sum(output_jig_2["pred_masks"] * torch.log2(output_jig_2["pred_masks"] + 1e-12), dim=1, keepdim=True),
            torch.sum(output_jig_4["pred_masks"] * torch.log2(output_jig_4["pred_masks"] + 1e-12), dim=1,
                      keepdim=True)], dim=1) / temperature)
        output_weight = output_weight.detach()
        pseudo_pred = output_weight[:, 0, :, :] * output_c["pred_masks"] + \
                      output_weight[:, 1, :, :] * output_jig_2["pred_masks"] + \
                      output_weight[:, 2, :, :] * output_jig_4["pred_masks"]
        pseudo_label = torch.argmax(pseudo_pred.detach(), dim=1, keepdim=False).unsqueeze(1)

        pred = pseudo_pred
        predictions_original_list = []
        for i in range(pred.shape[0]):
            prediction = np.uint8(np.argmax(pred[i, :, :, :].detach().cpu(), axis=0))
            prediction = keep_largest_connected_components(prediction)
            prediction = torch.from_numpy(prediction).to(device)
            predictions_original_list.append(prediction)
        predictions = torch.stack(predictions_original_list)
        predictions = torch.unsqueeze(predictions, 1)
        pred_keep_largest_connected = to_onehot_dim4(predictions, device)
        # -------------------------------------------------------
        # loss
        # -------------------------------------------------------
        weight_dict = criterion.weight_dict

        loss_dict = criterion(output_c, label)
        loss_scribble = sum(loss_dict[k] * weight_dict[k] for k in ['loss_CrossEntropy'] if k in weight_dict)
        loss_scribble = 1.0 * loss_scribble

        loss_scribble_jig_2 = criterion(output_jig_2, label)
        loss_scribble_jig_2 = sum(
            loss_scribble_jig_2[k] * weight_dict[k] for k in ['loss_CrossEntropy'] if k in weight_dict)
        loss_scribble_jig_2 = 1.0 * loss_scribble_jig_2

        loss_scribble_jig_4 = criterion(output_jig_4, label)
        loss_scribble_jig_4 = sum(
            loss_scribble_jig_4[k] * weight_dict[k] for k in ['loss_CrossEntropy'] if k in weight_dict)
        loss_scribble_jig_4 = 0.2 * loss_scribble_jig_4

        MSE_loss = nn.MSELoss(reduction='none')
        pseudo_foreground = torch.sum(pseudo_pred[:, 1:, :], dim=1, keepdim=True)
        edge_mask = torch.where(edge > 0.5, 1.0, 0.0)
        loss_edge = MSE_loss(pseudo_foreground * edge_mask, edge * edge_mask)
        loss_edge = (torch.sum(loss_edge, dim=(1, 2, 3)) / torch.sum(edge_mask, dim=(1, 2, 3))).mean() / 10

        loss_consistency_1_2 = 1 - Func.cosine_similarity(output_jig_2["pred_masks"], output_c["pred_masks"],
                                                          dim=1).mean()
        loss_consistency_1_2 = 0.3 * loss_consistency_1_2

        loss_consistency_1_4 = 1 - Func.cosine_similarity(output_jig_4["pred_masks"], output_c["pred_masks"],
                                                          dim=1).mean()
        loss_consistency_1_4 = 0.1 * loss_consistency_1_4

        loss_integrity = 1 - Func.cosine_similarity(pred[:, 0:4, :, :], pred_keep_largest_connected, dim=1).mean()
        loss_integrity = 0.3 * loss_integrity

        dice_loss = pDLoss(4, ignore_index=4)
        loss_pseudo = 0.1 * (dice_loss(output_c["pred_masks"], pseudo_label) +
                             dice_loss(output_jig_2["pred_masks"], pseudo_label) +
                             dice_loss(output_jig_4["pred_masks"], pseudo_label))

        loss_edge = 0.3 * loss_edge
        # -------------------------------------------------------
        loss = loss_scribble + loss_scribble_jig_2 + loss_scribble_jig_4 + \
               loss_consistency_1_2 + loss_consistency_1_4 + \
               loss_pseudo + loss_integrity + loss_edge

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # -------------------------------------------------------
        if step == 0:
            sample_list.append(sample[0].detach())
            label_list.append(label.argmax(1, keepdim=True)[0].detach())
            output_list.append(output_c['pred_masks'].argmax(1, keepdim=True)[0].detach())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss)

        itertime = time.time() - start
        metric_logger.log_every(step, total_steps, datatime, itertime, print_freq, header)
        step = step + 1

    # gather the stats from all processes
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / total_steps))
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    visual_train = Visualize_train()

    writer.add_scalar('loss', loss.item(), epoch)

    visual_train(torch.stack(sample_list), torch.stack(label_list), torch.stack(output_list), epoch, writer)
    return stats
