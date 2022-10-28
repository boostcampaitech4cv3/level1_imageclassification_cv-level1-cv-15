from yacs.config import CfgNode as CN

# Config Definition
_C = CN()

# Train
_C.train = True

# Model Hyperparams

_C.seed = 42
_C.epochs = 1
_C.dataset = "MaskSplitByProfileDataset"
_C.augmentation = "CustomAugmentation"
_C.cropsize = [320, 256]
_C.resize = [128, 96]
_C.batch_size = 64
_C.model = "BaseModel"


# Training
_C.optimizer = 'SGD'
_C.lr = 1e-3
_C.val_ratio = 0.2
_C.criterion = 'cross_entropy'
_C.lr_decay_step = 20
_C.log_interval = 20
_C.name = 'exp'

# Loss

_C.sampler = 'triplet' # triplet, triplet hard
_C.num_instance = 4
_C.loss_type = 'softmax_triplet'
_C.ID_loss_weight = 0.5
_C.triplet_loss_weight = 0.5
_C.label_smooth = False
_C.margin = 0.3
_C.feat_norm = False

# Validation

_C.validation_interval = 10
_C.valid_batch_size = 64

# Test 
_C.test_data_dir = "/opt/ml/input/eval"
_C.test_batch_size = 1000
_C.test_model = 'exp'

# Container enviornment

_C.data_dir = "/opt/ml/input/data/train/images"
_C.model_dir = "./model"
_C.output_dir = "./output"
