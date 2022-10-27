# pstage_01_image_classification

## Working Process with Time line
---

### 성수
- 1027
    - gitignore 추가
    - code 설명을 위한 주석 및 수정
        - dataset.py 의 MaskSplitByProfileDataset method
            ``` python 
            def split_dataset(self) -> List[Subset]:
                train_set = Subset(self,self.indices['train'])
                val_set = Subset(self,self.indices['val'])
                return train_set, val_set
            ```
        - train.py 의 save_directory 생성 및 config file dump
            ``` python
            # 위의 save_dir 폴더 만들기 
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            
            # 현재 arguments값을 config.json파일로 dump하기(나중에 hyperparameter값을 알기 위해)
            with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
                json.dump(vars(args), f, ensure_ascii=False, indent=4)
            ```
        - model.py 의 Mymodel class 예시 코드 (resnet34) 추가
### 민수
-
### 윤재
-
### 우택 
-

---

## Getting Started    
### Dependencies
- torch==1.7.1
- torchvision==0.8.2                                                              

### Install Requirements
- `pip install -r requirements.txt`

### Training
- `SM_CHANNEL_TRAIN={YOUR_TRAIN_IMG_DIR} SM_MODEL_DIR={YOUR_MODEL_SAVING_DIR} python train.py`

### Inference
- `SM_CHANNEL_EVAL={YOUR_EVAL_DIR} SM_CHANNEL_MODEL={YOUR_TRAINED_MODEL_DIR} SM_OUTPUT_DATA_DIR={YOUR_INFERENCE_OUTPUT_DIR} python inference.py`

### Evaluation
- `SM_GROUND_TRUTH_DIR={YOUR_GT_DIR} SM_OUTPUT_DATA_DIR={YOUR_INFERENCE_OUTPUT_DIR} python evaluation.py`