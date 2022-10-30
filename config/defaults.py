from yacs.config import CfgNode as CN

# Config Definition
_C = CN()

# Train
_C.train = True

# Model Hyperparams

_C.seed = 42
_C.dataset = "MaskSplitByProfileDataset"
_C.augmentation = "CustomAugmentation"
_C.cropsize = [320, 256]
_C.resize = [128, 96]
_C.batch_size = 64
_C.model = "BaseModel"


# Training

_C.val_ratio = 0.2
_C.criterion = 'cross_entropy'
_C.lr_decay_step = 20
_C.log_interval = 20
_C.name = 'exp'

# Optimizer

# Name of optimizer : SGD, AdamW
_C.optimizer = "Adam" 
# Name of scheduler
_C.scheduler = "cos"
# Number of max epoches
_C.epochs = 100
# Base learning rate
_C.base_lr = 3e-4
# Factor of learning bias
_C.bias_lr_factor = 1
# Factor of learning bias
_C.solver_seed = 1234
# Momentum
_C.momentum = 0.9
# Settings of weight decay
_C.weight_decay = 0.0005
_C.weight_decay_bias = 0.0005

# decay rate of learning rate
_C.gamma = 0.1
# decay step of learning rate
_C.steps = (40, 70)
# warm up factor
_C.warmup_factor = 0.01
# warm up epochs   
_C.warmup_epochs = 5
# method of warm up, option: 'constant','linear'
_C.warmup_method = "linear"

_C.cosine_margin = 0.5
_C.cosine_scale = 30

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

# WANDB

_C.wandb = False
# Container enviornment

_C.data_dir = "/opt/ml/input/data/train/images"
_C.model_dir = "./model_output"
_C.output_dir = "./inference_output"
