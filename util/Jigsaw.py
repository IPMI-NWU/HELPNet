import math
import random
import numpy as np
import torch
from torch import nn


def RandomBrightnessContrast(img, brightness_limit=0.2, contrast_limit=0.2, p=0.5):
    output = torch.zeros_like(img)
    threshold = 0.5

    for i in range(output.shape[0]):
        img_min, img_max = torch.min(img[i]), torch.max(img[i])

        output[i] = (img[i] - img_min) / (img_max - img_min) * 255.0
        if random.random() < p:
            brightness = 1.0 + random.uniform(-brightness_limit, brightness_limit)
            output[i] = torch.clamp(output[i] * brightness, 0., 255.)

            contrast = 0.0 + random.uniform(-contrast_limit, contrast_limit)
            output[i] = torch.clamp(output[i] + (output[i] - threshold * 255.0) * contrast, 0., 255.)

        output[i] = output[i] / 255.0 * (img_max - img_min) + img_min
    return output


class Jigsaw(nn.Module):
    def __init__(self):
        super(Jigsaw, self).__init__()

    def forward(self, imgs, num_x, num_y, shuffle_index=None):
        split_w, split_h = int(imgs.shape[2] / num_x), int(imgs.shape[3] / num_y)
        out_imgs = torch.zeros_like(imgs)
        imgs = imgs.unsqueeze(0)

        patches = torch.split(imgs, split_w, dim=3)
        patches = [torch.split(p, split_h, dim=4) for p in patches]
        patches = torch.cat([torch.cat(p, dim=0) for p in patches], dim=0)
        # ----------------------
        if shuffle_index is None:
            shuffle_index = np.random.permutation(num_x * num_y)
        else:
            shuffle_index = list(shuffle_index)
            shuffle_index = [shuffle_index.index(i) for i in range(num_x * num_y)]
        patches = patches[shuffle_index]

        x_index, y_index = 0, 0
        for patch in patches:
            out_imgs[:, :, y_index:y_index + split_h, x_index:x_index + split_w] = patch
            x_index += split_w
            if x_index == out_imgs.shape[2]:
                x_index = 0
                y_index += split_h
        return out_imgs, shuffle_index


class Jigsaw_4_4_keep2(nn.Module):
    def __init__(self):
        super(Jigsaw_4_4_keep2, self).__init__()
        self.jig_fun = Jigsaw()

    def forward(self, imgs ,num_x, num_y, shuffle_index=None):
        split_w, split_h = int(imgs.shape[2] / num_x), int(imgs.shape[3] / num_y)
        out_imgs = torch.zeros_like(imgs)
        imgs = imgs.unsqueeze(0)

        patches = torch.split(imgs, split_w, dim=3)
        patches = [torch.split(p, split_h, dim=4) for p in patches]
        patches = torch.cat([torch.cat(p, dim=0) for p in patches], dim=0)
        index_list = []
        patches_jig = []

        if shuffle_index is None:
            for i in range(len(patches)):
                p_jig, index = self.jig_fun(patches[i], 2, 2)
                patches_jig.append(p_jig)
                index_list.append(index)
        else:
            for i in range(len(patches)):
                p_jig, _ = self.jig_fun(patches[i], 2, 2, shuffle_index[i])
                patches_jig.append(p_jig)

        x_index, y_index = 0, 0
        for patch in patches_jig:
            out_imgs[:, :, y_index:y_index + split_h, x_index:x_index + split_w] = patch
            x_index += split_w
            if x_index == out_imgs.shape[2]:
                x_index = 0
                y_index += split_h
        return out_imgs, index_list


class LocalJigsaw(nn.Module):
    def __init__(self, img_size, local_size):
        super(LocalJigsaw, self).__init__()
        self.img_size = img_size
        self.local_size = local_size
        self.index_list = []
        self.center_coord_list = []
        self.jigsaw_fun = Jigsaw()
        self.patch_num = int(math.sqrt(2 * 2))

    def jigsaw_aug(self, img):
        x_coord = random.randint(0, self.img_size - self.local_size)
        y_coord = random.randint(0, self.img_size - self.local_size)

        output = torch.zeros_like(img)

        local_area = img[:, :, x_coord: x_coord + self.local_size, y_coord: y_coord + self.local_size]
        local_area_jigsaw, index = self.jigsaw_fun(local_area, self.patch_num, self.patch_num)
        output[:, :, x_coord: x_coord + self.local_size, y_coord: y_coord + self.local_size] = local_area_jigsaw
        output[:, :, 0: x_coord, y_coord: y_coord + self.local_size] = img[:, :, 0: x_coord, y_coord: y_coord + self.local_size]
        output[:, :, x_coord + self.local_size:, y_coord: y_coord + self.local_size] = img[:, :, x_coord + self.local_size:, y_coord: y_coord + self.local_size]
        output[:, :, :, 0:y_coord] = img[:, :, :, 0:y_coord]
        output[:, :, :, y_coord + self.local_size:] = img[:, :, :, y_coord + self.local_size:]

        self.center_coord_list.append([x_coord, y_coord])
        self.index_list.append(index)
        return output

    def restoration(self, img):
        if len(self.index_list) == 0:
            return
        x_coord, y_coord = self.center_coord_list[-1][0], self.center_coord_list[-1][1]
        local_area = img[:, :, x_coord: x_coord + self.local_size, y_coord: y_coord + self.local_size]
        index = self.index_list[-1]
        local_area_res, _ = self.jigsaw_fun(local_area, self.patch_num, self.patch_num, index)
        # img[:, :, x_coord: x_coord + self.local_size, y_coord: y_coord + self.local_size] = local_area_res

        output = torch.zeros_like(img)
        output[:, :, x_coord: x_coord + self.local_size, y_coord: y_coord + self.local_size] = local_area_res
        output[:, :, 0: x_coord, y_coord: y_coord + self.local_size] = img[:, :, 0: x_coord, y_coord: y_coord + self.local_size]
        output[:, :, x_coord + self.local_size:, y_coord: y_coord + self.local_size] = img[:, :, x_coord + self.local_size:, y_coord: y_coord + self.local_size]
        output[:, :, :, 0:y_coord] = img[:, :, :, 0:y_coord]
        output[:, :, :, y_coord + self.local_size:] = img[:, :, :, y_coord + self.local_size:]

        self.index_list.pop()
        self.center_coord_list.pop()
        return output


class SwinJigsaw(nn.Module):
    def __init__(self, img_size, shift_size):
        super(SwinJigsaw, self).__init__()
        self.choices = [1, -1]
        self.img_size = img_size
        self.shift_size = shift_size
        self.x_shift = 0
        self.y_shift = 0
        self.square_shuffle_index = None
        self.rectangular_shuffle_index = None

    def swinjigsaw_aug(self, img):
        imgsz = self.img_size
        shiftsz = self.shift_size
        out_img = torch.zeros_like(img)
        img = img.unsqueeze(0)

        square_list = [img[:, :, :, 0: shiftsz, 0: shiftsz],
                       img[:, :, :, (imgsz - shiftsz):, 0: shiftsz],
                       img[:, :, :, 0: shiftsz, (imgsz - shiftsz):],
                       img[:, :, :, (imgsz - shiftsz):, (imgsz - shiftsz):]]

        square = torch.cat(square_list, dim=0)
        square_shuffle_index = list(np.random.permutation(4))
        self.square_shuffle_index = square_shuffle_index
        square = square[square_shuffle_index]

        out_img[:, :, 0: shiftsz, 0: shiftsz] = square[0]
        out_img[:, :, (imgsz - shiftsz):, 0: shiftsz] = square[1]
        out_img[:, :, 0: shiftsz, (imgsz - shiftsz):] = square[2]
        out_img[:, :, (imgsz - shiftsz):, (imgsz - shiftsz):] = square[3]

        rectangular_list = [img[:, :, :, shiftsz:(imgsz - shiftsz), 0: shiftsz],
                       img[:, :, :, (imgsz - shiftsz):, shiftsz:(imgsz - shiftsz)].transpose(3, 4),
                       img[:, :, :, shiftsz:(imgsz - shiftsz), (imgsz - shiftsz):],
                       img[:, :, :, 0: shiftsz, shiftsz:(imgsz - shiftsz)].transpose(3, 4)]

        rectangular = torch.cat(rectangular_list, dim=0)
        rectangular_shuffle_index = list(np.random.permutation(4))
        self.rectangular_shuffle_index = rectangular_shuffle_index
        rectangular = rectangular[rectangular_shuffle_index]

        out_img[:, :, shiftsz:(imgsz - shiftsz), 0: shiftsz] = rectangular[0]
        out_img[:, :, (imgsz - shiftsz):, shiftsz:(imgsz - shiftsz)] = rectangular[1].transpose(2, 3)
        out_img[:, :, shiftsz:(imgsz - shiftsz), (imgsz - shiftsz):] = rectangular[2]
        out_img[:, :, 0: shiftsz, shiftsz:(imgsz - shiftsz)] = rectangular[3].transpose(2, 3)

        out_img[:, :, shiftsz:(imgsz - shiftsz), shiftsz:(imgsz - shiftsz)] = img[:, :, :, shiftsz:(imgsz - shiftsz), shiftsz:(imgsz - shiftsz)]

        self.x_shift, self.y_shift = random.choice(self.choices), random.choice(self.choices)
        out_img = torch.roll(out_img, shifts=(self.x_shift * shiftsz, self.y_shift * shiftsz), dims=(2, 3))

        return out_img


    def restoration(self, img):
        imgsz = self.img_size
        shiftsz = self.shift_size
        img = torch.roll(img, shifts=(-1 * self.x_shift * shiftsz, -1 * self.y_shift * shiftsz), dims=(2, 3))
        out_img = torch.zeros_like(img)
        img = img.unsqueeze(0)

        square_list = [img[:, :, :, 0: shiftsz, 0: shiftsz],
                       img[:, :, :, (imgsz - shiftsz):, 0: shiftsz],
                       img[:, :, :, 0: shiftsz, (imgsz - shiftsz):],
                       img[:, :, :, (imgsz - shiftsz):, (imgsz - shiftsz):]]

        square = torch.cat(square_list, dim=0)
        square_shuffle_index = list(self.square_shuffle_index)
        square_shuffle_index = [square_shuffle_index.index(i) for i in range(4)]
        square = square[square_shuffle_index]

        out_img[:, :, 0: shiftsz, 0: shiftsz] = square[0]
        out_img[:, :, (imgsz - shiftsz):, 0: shiftsz] = square[1]
        out_img[:, :, 0: shiftsz, (imgsz - shiftsz):] = square[2]
        out_img[:, :, (imgsz - shiftsz):, (imgsz - shiftsz):] = square[3]

        rectangular_list = [img[:, :, :, shiftsz:(imgsz - shiftsz), 0: shiftsz],
                            img[:, :, :, (imgsz - shiftsz):, shiftsz:(imgsz - shiftsz)].transpose(3, 4),
                            img[:, :, :, shiftsz:(imgsz - shiftsz), (imgsz - shiftsz):],
                            img[:, :, :, 0: shiftsz, shiftsz:(imgsz - shiftsz)].transpose(3, 4)]

        rectangular = torch.cat(rectangular_list, dim=0)
        rectangular_shuffle_index = list(self.rectangular_shuffle_index)
        rectangular_shuffle_index = [rectangular_shuffle_index.index(i) for i in range(4)]
        rectangular = rectangular[rectangular_shuffle_index]

        out_img[:, :, shiftsz:(imgsz - shiftsz), 0: shiftsz] = rectangular[0]
        out_img[:, :, (imgsz - shiftsz):, shiftsz:(imgsz - shiftsz)] = rectangular[1].transpose(2, 3)
        out_img[:, :, shiftsz:(imgsz - shiftsz), (imgsz - shiftsz):] = rectangular[2]
        out_img[:, :, 0: shiftsz, shiftsz:(imgsz - shiftsz)] = rectangular[3].transpose(2, 3)

        out_img[:, :, shiftsz:(imgsz - shiftsz), shiftsz:(imgsz - shiftsz)] = img[:, :, :, shiftsz:(imgsz - shiftsz),
                                                                              shiftsz:(imgsz - shiftsz)]
        return out_img
