import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import torch.nn.functional as F
import imgaug as ia
import imgaug.augmenters as iaa


def mask_to_onehot(mask, ):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    mask = np.expand_dims(mask,-1)
    for colour in range (9):
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.int32)
    return semantic_map

def augment_seg(img_aug, img, seg ):
    seg = mask_to_onehot(seg)
    aug_det = img_aug.to_deterministic()
    image_aug = aug_det.augment_image(img)

    segmap = ia.SegmentationMapOnImage(seg, nb_classes=np.max(seg)+1, shape=img.shape)
    segmap_aug = aug_det.augment_segmentation_maps( segmap)
    segmap_aug = segmap_aug.get_arr_int()
    segmap_aug = np.argmax(segmap_aug, axis=-1).astype(np.float32)
    return image_aug, segmap_aug


#图像增强（旋转、翻转、复制）
def random_rot_flip(image, label):
    k = np.random.randint(0, 4) #生成0-3之间的整数
    image = np.rot90(image, k)  #对图像和标签进行k*90的旋转
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()  #进行沿指定轴的翻转（0是水平轴，1是垂直轴）
    label = np.flip(label, axis=axis).copy()
    return image, label

#对图像和标签执行随机旋转
def random_rotate(image, label):
    angle = np.random.randint(-20, 20) #旋转的角度
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

#图像数据增强生成器
class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample): #sample字典
        image, label = sample['image'], sample['label']
        #决定执行随机旋转和翻转还是仅进行随机旋转
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        # print(x, y)     -----(512,512)
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?对图像使用三次插值
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)  # 对标签使用最近邻插值
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)#转换为张量，对图像添加一个批次维度
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None, img_size=224):   #这个图片尺寸是新添加的
        self.transform = transform  # using transform in torch!
        self.split = split  #指定数据集划分，如train或test
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines() #读取划分的文本文件
        self.data_dir = base_dir   #data_dir=base_dir=root_path


        #下面这个是新添加的
        self.img_size = img_size

        self.img_aug = iaa.SomeOf((0, 4), [
            iaa.Flipud(0.5, name="Flipud"),
            iaa.Fliplr(0.5, name="Fliplr"),
            iaa.AdditiveGaussianNoise(scale=0.005 * 255),
            iaa.GaussianBlur(sigma=1.0),
            iaa.LinearContrast((0.5, 1.5), per_channel=0.5),
            iaa.Affine(scale={"x": (0.5, 2), "y": (0.5, 2)}),
            iaa.Affine(rotate=(-40, 40)),
            iaa.Affine(shear=(-16, 16)),
            iaa.PiecewiseAffine(scale=(0.008, 0.03)),
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
        ], random_order=True)





    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path)
            # print(data["image"][0])  #['image','label']，每个键对应的是一个数组，数组的每一项又是一个数组
            image, label = data['image'], data['label']  #每个样本包含image和label两个键
            # print(data["label"])  #['image','label']，每个键对应的是一个数组，数组的每一项又是一个数组




            #下面这个是新添加的
            image, label = augment_seg(self.img_aug, image, label)
            x, y = image.shape
            if x != self.img_size or y != self.img_size:
                image = zoom(image, (self.img_size / x, self.img_size / y), order=3)
                label = zoom(label, (self.img_size / x, self.img_size / y), order=0)

        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n') #将样本名称添加到sample字典中
        return sample





