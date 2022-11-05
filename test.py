import os
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE 
from importlib import import_module
import numpy as np
import torch
from time import time
import tqdm
import matplotlib.pyplot as plt

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

MASK = 0 , INCORRECT = 1, NORMAL = 2
MALE = 0 , FEMALE = 1,
YONG = 0, MIDDLE = 1, OLD =2 
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

start=time()


X, y = make_classification(n_classes=18, class_sep=2,
weights=[0.14357479533614487, 0.10543289506325974, 0.017055321260233194,
 0.186988340362193, 0.20869511287521708, 0.024187546514512527, 
 0.028714959067228974, 0.021086579012651947, 0.0034110642520466384, 
 0.0373976680724386, 0.041739022575043416, 0.004837509302902505, 
 0.028714959067228974, 0.021086579012651947, 0.046824609278094766, 
 0.0373976680724386, 0.041739022575043416, 0.001116348300669809], 
 n_informative=5, n_redundant=1, flip_y=0,
n_features=20, n_clusters_per_class=1, n_samples=16124, random_state=10)

print(type(X),type(y))
print(X.shape,y.shape)


print('Original dataset shape %s' % Counter(y))
# # Original dataset shape Counter({1: 900, 0: 100})

sm = SMOTE(sampling_strategy = 'auto', random_state=42)
X_res, y_res = sm.fit_resample(X, y)

print(type(X_res),type(y_res))
print(X_res.shape,y_res.shape)
print(y_res.reshape(-1,1).shape)


print('Resampled dataset shape %s' % Counter(y_res))
# Resampled dataset shape Counter({0: 900, 1: 900})


# -- dataset
dataset_module = getattr(import_module("dataset"), 'MaskSplitByProfileDataset')  # default: MaskBaseDataset
dataset = dataset_module(
    data_dir=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'),
)
num_classes = dataset.num_classes  # 18

# -- augmentation
transform_module = getattr(import_module("dataset"), 'CustomAugmentation')  # default: BaseAugmentation
transform = transform_module(
    resize=(384,384),
    mean=dataset.mean,
    std=dataset.std,
)
dataset.set_transform(transform)

print(len(dataset))
# print(dataset[0][0].shape)
# print(np.array([1,2,3,4]).reshape(-1,1).shape)
# t1,t2 = zip([1,"a"],[2,"b"])
# print(type(np.array(t1)),t2)

print("-"*50 + "Starting unpacking" + "-"*50)
data,label = zip(*dataset)
data = list(data)
label = np.array(label).reshape(-1,1)
print(data)
print(label)


# print(data.shape)
# print(label.shape)
# # print(data[0],label[0])


# data_res, label_res = sm.fit_resample(data,label)

# print('Resampled dataset shape %s' % Counter(label_res))

end=time()

print(end-start)

print("-"*50 + "Ended" + "-"*50)




