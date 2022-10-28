import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os
args=None
df=pd.read_csv('input/data/train/train.csv')

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]
IMG_HEAD=[
    'mask1','mask2','mask3','mask4','mask5','incorrect_mask','normal'
]

for i in range(len(df)):
    gender_label=0
    age_label=0
    gender=df.at[i,'gender']
    age=df.at[i,'age']

    if 30<=df.at[i,'age']<60:
        age_label=2
    elif df.at[i,'age']>=60:
        age_label=4
    if df.at[i,'gender']=='female':
        gender_label=1
    
    df.loc[i,'label']=age_label+gender_label


kf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
for fold,(train_idx,val_idx) in enumerate(kf.split(range(len(df)),y=df['label'])):
    df.loc[val_idx,'fold']=fold

k_fold_df=pd.DataFrame(columns=['path','label','fold'])

for i in range(len(df)):
    gender_label=0
    age_label=0

    if 30<=df.at[i,'age']<60:
        age_label=1
    elif df.at[i,'age']>=60:
        age_label=2
    if df.at[i,'gender']=='female':
        gender_label=3

    folder_name=os.path.join('input/data/train/images',df.at[i,'path'])
    for file in os.listdir(folder_name):
        head,ext=os.path.splitext(file)
        if head in IMG_HEAD:
            mask_label=0
            if 'incorrect' in head:
                mask_label=6
            elif 'normal' in head:
                mask_label=12

            path=os.path.join(folder_name,file)
            label=gender_label+mask_label+age_label
            fold=df.at[i,'fold']
            
            k_fold_df=k_fold_df.append(pd.DataFrame([[path,label,fold]],columns=['path','label','fold']))

print(k_fold_df)
k_fold_df.to_csv('input/data/train/kfold.csv',index=False)
            
           
            
           