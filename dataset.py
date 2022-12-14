import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List
from torch.utils.data import Dataset, Subset, random_split
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import *
import pandas as pd
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2

from sklearn.model_selection import StratifiedKFold

'''
Class Info
Class |  Mask      | Gender   | Age             | Sample cnt
0     |  Wear      | Male     | <30             | 2315
1     |  Wear      | Male     | >=30 and < 60   | 1700
2     |  Wear      | Male     | >= 60           | 275 *
3     |  Wear      | Female   | <30             | 3015
4     |  Wear      | Female   | >=30 and < 60   | 3365
5     |  Wear      | Female   | >= 60           | 390 *
6     |  Incorrect | Male     | <30             | 463
7     |  Incorrect | Male     | >=30 and < 60   | 340
8     |  Incorrect | Male     | >= 60           | 55 *
9     |  Incorrect | Female   | <30             | 603
10    |  Incorrect | Female   | >=30 and < 60   | 673
11    |  Incorrect | Female   | >= 60           | 78 *
12    |  Not wear  | Male     | <30             | 463
13    |  Not wear  | Male     | >= 30 and < 60  | 340
14    |  Not wear  | Male     | >= 60           | 755 *
15    |  Not wear  | Female   | <30             | 603 
16    |  Not wear  | Female   | >=30 and < 60   | 673
17    |  Not wear  | Female   | >= 60           | 18 *

'''

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


class AddGaussianNoise(object):
    """
        transform ??? ?????? ???????????? ??????????????? __init__, __call__, __repr__ ?????????
        ?????? ???????????? ????????? ??? ????????????.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = A.Compose([
            A.HorizontalFlip(),
            A.CenterCrop(320, 256),
            A.Resize(380,380),
            A.ColorJitter(0.1, 0.1, 0.1, 0.1),
            #A.CLAHE(always_apply=False, p=0.5, clip_limit=(1, 15), tile_grid_size=(8, 8)),
            A.Equalize(always_apply=False, p=0.5, mode='cv', by_channels=False),
            A.CoarseDropout(always_apply=False, p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    def __call__(self, image):
        return self.transform(image=image)['image']


class fold_mask(Dataset):
    num_classes=18
    def __init__(self,img_paths,labels,mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246),transforms=None):
        self.img_paths=img_paths
        self.labels=labels
        self.transforms=transforms
        self.mean=mean
        self.std=std

    def set_transform(self, transform):
        self.transforms = transform
        
    def __getitem__(self,index):
        img_path=self.img_paths[index]
        image=cv2.imread(img_path)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        if self.transforms is not None:
            image=self.transforms(image)
        label=self.labels[index]
        return image,label

    def __len__(self):
        return len(self.img_paths)


class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 29:
            return cls.YOUNG
        elif value < 57:
            return cls.MIDDLE
        else:
            return cls.OLD

class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []
    train_image_with_ID = []
    val_image_with_ID = []
    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio

        self.transform = None
        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." ??? ???????????? ????????? ???????????????
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." ??? ???????????? ?????? ??? invalid ??? ???????????? ???????????????
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform ???????????? ???????????? transform ??? ??????????????????"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        image_transform = self.transform(image)
        return image_transform, multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        image=cv2.imread(image_path)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        return image

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """
        ??????????????? train ??? val ??? ????????????,
        pytorch ????????? torch.utils.data.random_split ????????? ????????????
        torch.utils.data.Subset ????????? ?????? ????????????.
        ????????? ????????? ????????? ????????? ?????? IDE (e.g. pycharm) ??? navigation ????????? ?????? ????????? ??? ??? ???????????? ?????? ??????????????????^^
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set


class MaskSplitByProfileDataset(MaskBaseDataset):
    """
        train / val ????????? ????????? ???????????? ????????? random ??? ??????
        ??????(profile)??? ???????????? ????????????.
        ????????? val_ratio ??? ?????? train / val ????????? ?????? ????????? ????????? ?????? ??????(profile)??? ????????? ???????????? indexing ??? ?????????
        ?????? `split_dataset` ?????? index ??? ?????? Subset ?????? dataset ??? ???????????????.
    """

    def __init__(self, data_dir, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.choices(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {
            "train": train_indices,
            "val": val_indices
        }

    def setup(self):
        '''
        df=pd.read_csv('input/data/train/sample_train.csv')
        profiles=df['path'].values


        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt=0
        for (phase, indices) in split_profiles.items():

            #mask_sampled_2=0
            #mask_sampled_3=0
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:  # "." ??? ???????????? ?????? ??? invalid ??? ???????????? ???????????????
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)
                    
                    if phase=='train':
                        if int(age)==59:
                            continue
                    #    if int(age)>=50 and int(age)<60 and mask_sampled_2<3:
                     #       mask_sampled_2+=1
                      #      continue
                    

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1


        '''
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:  # "." ??? ???????????? ?????? ??? invalid ??? ???????????? ???????????????
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    # if phase=='train':                  
                    #     if gender_label==1 and 56<=int(age)<=59 and ('mask4' in _file_name or  'mask1' in _file_name or 'mask5' in _file_name or 'mask2' in _file_name):
                    #         continue

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)
                    if phase == 'train':
                        self.train_image_with_ID.append((img_path,mask_label * 6 + gender_label * 3 + age_label))
                    else:
                        
                        self.val_image_with_ID.append((img_path,mask_label * 6 + gender_label * 3 + age_label))
                    self.indices[phase].append(cnt)
                    cnt += 1
        self.train_image_with_ID = sorted(self.train_image_with_ID, key = lambda x : x[1])
        self.val_image_with_ID= sorted(self.val_image_with_ID, key = lambda x : x[1])
    
    def split_dataset(self) -> List[Subset]:

            return [Subset(self, indices) for phase, indices in self.indices.items()]

    
class TestDataset(Dataset):
    def __init__(self, img_paths, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        self.transform = A.Compose([
            #A.HorizontalFlip(),
            A.CenterCrop(320, 256),
            A.Resize(380,380),
            #A.ColorJitter(0.1, 0.1, 0.1, 0.1),
            #A.CLAHE(always_apply=False, p=0.5, clip_limit=(1, 15), tile_grid_size=(8, 8)),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    def __call__(self, image):
        return self.transform(image=image)['image']

    def __getitem__(self, index):
        image = cv2.imread(self.img_paths[index])
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']
        return image

    def __len__(self):
        return len(self.img_paths)

class ImageDataset(Dataset): # ImageDataset class : DataLoader??? ????????? instance ????????? __len__, _getitem__ method 
    def __init__(self, dataset, transform=None):
        self.dataset = sorted(dataset,key=lambda x : x[1]) # dataset list 
        self.transform = transform # transform list
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image_path,pid = self.dataset[index]
        image=cv2.imread(image_path)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        if self.transform is not None:
             img = self.transform(image)
        return img, pid