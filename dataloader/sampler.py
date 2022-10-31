from torch.utils.data.sampler import Sampler
from collections import defaultdict
import collections
import copy
import random
import numpy as np

'''
Class Info
Class |  Mask      | Gender   | Age             | Sample cnt
0     |  Wear      | Male     | <30             | 2315
1     |  Wear      | Male     | >=30 and < 60   | 1700
2     |  Wear      | Male     | >= 60           | 275
3     |  Wear      | Female   | <30             | 3015
4     |  Wear      | Female   | >=30 and < 60   | 3365
5     |  Wear      | Female   | >= 60           | 390
6     |  Incorrect | Male     | <30             | 463
7     |  Incorrect | Male     | >=30 and < 60   | 340
8     |  Incorrect | Male     | >= 60           | 55
9     |  Incorrect | Female   | <30             | 603
10    |  Incorrect | Female   | >=30 and < 60   | 673
11    |  Incorrect | Female   | >= 60           | 78
12    |  Not wear  | Male     | <30             | 463
13    |  Not wear  | Male     | >= 30 and < 60  | 340
14    |  Not wear  | Male     | >= 60           | 755
15    |  Not wear  | Female   | <30             | 603 
16    |  Not wear  | Female   | >=30 and < 60   | 673
17    |  Not wear  | Female   | >= 60           | 18

'''

def SampleWeighting(batch_idxs_dict,under_id,under_rate,over_id,over_rate):
    
    # Undersample
    for id in under_id:
        batch_length = len(batch_idxs_dict[id])
        batch_idxs_dict[id] =  batch_idxs_dict[id][: batch_length // under_rate]
    # Oversample
    # for id in over_id:
    #     batch_idxs_dict[id] = over_rate * batch_idxs_dict[id]
    
    return batch_idxs_dict

class RandomIdentitySampler(Sampler):  # For Single GPU training
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid).
    """
    def __init__(self, data_source,cfg):
        self.data_source = sorted(data_source,key=lambda x : x[1]) # Imagedataset return : [('image_dir',pid,camid,trackid')]
        self.batch_size = cfg.batch_size
        self.num_instances = cfg.num_instance
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) # dict with list value
        self.undersample_id = cfg.undersample_id
        self.oversample_id = cfg.oversample_id
        self.undersample_rate = cfg.undersample_rate
        self.oversample_rate = cfg.oversample_rate

        
        for index, (_,pid) in enumerate(self.data_source):
            self.index_dic[pid].append(index) # pids : image index 형태의 dictionary로 저장
        
        # self.index_dic = dict(collections.OrderedDict(sorted(self.index_dic.items())))
        self.id_cnt = list(map(lambda x : len(x[1]),self.index_dic.items()))
        self.pids = list(self.index_dic.keys())  # pids list

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids: # pid 들에 대해
            idxs = self.index_dic[pid] # pid에 해당하는 image index들 
            num = len(idxs)
            if num < self.num_instances: # 만약 해당 id의 image 개수가 num_instances보다 작으면 전부 batch에 포함
                num = self.num_instances
            self.length += num - num % self.num_instances  # number of examples 

    def __iter__(self):
        batch_idxs_dict = defaultdict(list) 

        for pid in self.pids: # pid들에 대해
            idxs = copy.deepcopy(self.index_dic[pid]) # pid에 해당하는 image index들
            if len(idxs) < self.num_instances: # 만약 해당 id의 image 개수가 num_instances보다 작으면
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True) # 중복을 혀용하여 random choose한 idx list 반환
            random.shuffle(idxs)
            batch_idxs = []
            # 현재 pid에 대해
            for idx in idxs: # random shuffle 된 idx에 대해
                batch_idxs.append(idx) # batch_idxs list에 idx를 추가
                if len(batch_idxs) == self.num_instances: # batch에 pid에 해당하는 image를 num_instances만큼 추가했다면, 
                    batch_idxs_dict[pid].append(batch_idxs) # batch_idxs_dict에 pid : batch 형태로 추가
                    batch_idxs = [] # batch 초기화
                    # 예를 들면, 0번 pid에 대해 {0 :[[26,40,41,34],[21,6,10,44]...], }
        
        batch_idxs_len = list(map(lambda x : len(x[1]),batch_idxs_dict.items()))
        
        # Over,Undersampling
        batch_idxs_dict = SampleWeighting(batch_idxs_dict,self.undersample_id,self.undersample_rate,self.oversample_id,self.oversample_rate)
        
        batch_idxs_len_sampled = list(map(lambda x : len(x[1]),batch_idxs_dict.items()))

        avai_pids = copy.deepcopy(self.pids) # pids
        final_idxs = []
        cnt = 0
        while len(avai_pids) >= self.num_pids_per_batch + len(self.oversample_id): # self.num_instances = 4, self.num_pids_per_batch = 16
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            cnt +=1
            # print(f"{cnt}th ID batch : {selected_pids}")
            for pid in selected_pids: # selected_pids : random으로 뽑힌 self.num_pids_per_batch만큼의 pid
                if pid in self.oversample_id:
                    temp = batch_idxs_dict[pid].pop(0)
                    batch_idxs_dict[pid] += [temp]
                    batch_idxs = temp
                else : 
                    batch_idxs = batch_idxs_dict[pid].pop(0) # self.num_instances개씩 뽑아놓은 batch들중 맨 앞을 pop
                final_idxs.extend(batch_idxs) # final_idxs에 batch_idxs elements들을 추가
                if len(batch_idxs_dict[pid]) == 0: # 해당 pid에 대해 더이상 추가할 sample들이 없으면
                    avai_pids.remove(pid) # id list에서 pid를 제거

        # while문이 끝나고 나면?
        # len(avai_pids) == self.num_pids_per_batch
        # batch_idxs_dict : avai_pids에 남은 16개의 pid를 제외한 모든 pid key의 value가 []인 상태
        
        # final_idxs 에는 random id에 대해 sample들이 4개씩 
        
        return iter(final_idxs)

    def __len__(self): 
        return self.length
