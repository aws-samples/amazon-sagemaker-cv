import os

from yacs.config import CfgNode as CN

_C = CN()

_C.LOG_INTERVAL = 50 # How frequently the runner will log information
_C.S3_INTERVAL = 100

_C.HOOKS = ["CheckpointHook", 
            "IterTimerHook", 
            "TextLoggerHook", 
            "TensorboardMetricsLogger",
            "CocoEvaluator"]

_C.PATHS = CN()
_C.PATHS.TRAIN_FILE_PATTERN = '/opt/ml/input/data/train/train*'
_C.PATHS.VAL_FILE_PATTERN = '/opt/ml/input/data/val/val*'
_C.PATHS.WEIGHTS = "/opt/ml/input/data/weights/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603"
_C.PATHS.VAL_ANNOTATIONS = '/opt/ml/input/data/annotations/instances_val2017.json'
_C.PATHS.OUT_DIR = "/opt/ml/checkpoints"

########################################################################
# Input contains information about our dataset
########################################################################

_C.INPUT = CN()
_C.INPUT.IMAGE_SIZE = (832, 1344)
_C.INPUT.NUM_CLASSES = 91
_C.INPUT.TRAIN_BATCH_SIZE = 32 # This is global batch size across all GPUs
_C.INPUT.EVAL_BATCH_SIZE = 32
_C.INPUT.DATALOADER = "CocoInputReader" 
_C.INPUT.VISUALIZE_IMAGES_SUMMARY = False
_C.INPUT.SKIP_CROWDS_DURING_TRAINING = True
_C.INPUT.INCLUDE_GROUNDTRUTH_IN_FEATURES = False
_C.INPUT.USE_CATEGORY = True
_C.INPUT.AUGMENT_INPUT_DATA = True
_C.INPUT.GT_MASK_SIZE = 112

########################################################################
# Parameters of the model
########################################################################

_C.MODEL = CN()
_C.MODEL.DETECTOR = "TwoStageDetector"
_C.MODEL.INCLUDE_MASK = True
_C.MODEL.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)

_C.MODEL.BACKBONE = CN()

_C.MODEL.BACKBONE.CONV_BODY = "resnet50"
_C.MODEL.BACKBONE.DATA_FORMAT = "channels_last"
_C.MODEL.BACKBONE.TRAINABLE = True
_C.MODEL.BACKBONE.FINETUNE_BN = False
_C.MODEL.BACKBONE.NORM_TYPE = "batchnorm"
_C.MODEL.BACKBONE.NECK = "FPN"

########################################################################
# RPN head
########################################################################

_C.MODEL.DENSE = CN()

_C.MODEL.DENSE.RPN_HEAD = "StandardRPNHead"
_C.MODEL.DENSE.FEAT_CHANNELS = 256
_C.MODEL.DENSE.TRAINABLE = True
_C.MODEL.DENSE.MIN_LEVEL = 2
_C.MODEL.DENSE.MAX_LEVEL = 6
_C.MODEL.DENSE.NUM_SCALES = 1
_C.MODEL.DENSE.ASPECT_RATIOS = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
_C.MODEL.DENSE.ANCHOR_SCALE = 8.0

_C.MODEL.DENSE.POSITIVE_OVERLAP = 0.7
_C.MODEL.DENSE.NEGATIVE_OVERLAP = 0.3
_C.MODEL.DENSE.BATCH_SIZE_PER_IMAGE = 256
_C.MODEL.DENSE.FG_FRACTION = 0.5

_C.MODEL.DENSE.LOSS_TYPE = "huber"
_C.MODEL.DENSE.LABEL_SMOOTHING = 0.0
_C.MODEL.DENSE.LOSS_WEIGHT = 1.0

_C.MODEL.DENSE.PRE_NMS_TOP_N_TRAIN = 2000
_C.MODEL.DENSE.PRE_NMS_TOP_N_TEST = 1000

_C.MODEL.DENSE.POST_NMS_TOP_N_TRAIN = 1000
_C.MODEL.DENSE.POST_NMS_TOP_N_TEST = 1000

_C.MODEL.DENSE.NMS_THRESH = 0.7
_C.MODEL.DENSE.MIN_SIZE = 0.

_C.MODEL.DENSE.USE_FAST_BOX_PROPOSAL = True
_C.MODEL.DENSE.USE_BATCHED_NMS = True

########################################################################
# RCNN head
########################################################################

_C.MODEL.RCNN = CN()
_C.MODEL.RCNN.ROI_HEAD = "StandardRoIHead"
_C.MODEL.RCNN.SAMPLER = "RandomSampler"
_C.MODEL.RCNN.BATCH_SIZE_PER_IMAGE = 512
_C.MODEL.RCNN.FG_FRACTION = 0.25
_C.MODEL.RCNN.THRESH = 0.5
_C.MODEL.RCNN.THRESH_HI = 0.5
_C.MODEL.RCNN.THRESH_LO = 0.0

_C.MODEL.FRCNN = CN()
_C.MODEL.FRCNN.BBOX_HEAD = "StandardBBoxHead"
_C.MODEL.FRCNN.ROI_EXTRACTOR = "GenericRoIExtractor"
_C.MODEL.FRCNN.ROI_SIZE = 7
_C.MODEL.FRCNN.GPU_INFERENCE = False
_C.MODEL.FRCNN.MLP_DIM = 1024
_C.MODEL.FRCNN.TRAINABLE = True
_C.MODEL.FRCNN.LOSS_TYPE = "huber"
_C.MODEL.FRCNN.LABEL_SMOOTHING = 0.0
_C.MODEL.FRCNN.CLASS_AGNOSTIC = False
_C.MODEL.FRCNN.CARL = False
_C.MODEL.FRCNN.LOSS_WEIGHT = 1.0

_C.MODEL.MRCNN = CN()
_C.MODEL.MRCNN.MASK_HEAD = "StandardMaskHead"
_C.MODEL.MRCNN.RESOLUTION = 28
_C.MODEL.MRCNN.ROI_EXTRACTOR = "GenericRoIExtractor"
_C.MODEL.MRCNN.ROI_SIZE = 14
_C.MODEL.MRCNN.GPU_INFERENCE = False
_C.MODEL.MRCNN.TRAINABLE = True
_C.MODEL.MRCNN.LOSS_WEIGHT = 1.0
_C.MODEL.MRCNN.LABEL_SMOOTHING = 0.0

########################################################################
# Solver
########################################################################

_C.SOLVER = CN()
_C.SOLVER.TRAINER = "DetectionTrainer"
_C.SOLVER.OPTIMIZER = "Momentum"
_C.SOLVER.LR = 0.04
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.NESTEROV = False
_C.SOLVER.BETA_1 = 0.9
_C.SOLVER.BETA_2 = 0.999
_C.SOLVER.EPSILON = 1.0e-7
_C.SOLVER.NUM_IMAGES = 118287 # number of images in COCO
_C.SOLVER.MAX_ITERS = 60000
_C.SOLVER.ALPHA = 0.02
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.GRADIENT_CLIP_RATIO = 0.0
_C.SOLVER.XLA = True
_C.SOLVER.FP16 = True
_C.SOLVER.TF32 = False
_C.SOLVER.SCHEDULE = "PiecewiseConstantDecay"
_C.SOLVER.DECAY_STEPS = [35000, 50000]
_C.SOLVER.DECAY_LR = [.1, .01]
_C.SOLVER.WARMUP = "LinearWarmup"
_C.SOLVER.WARM_UP_RATIO = 0.1
_C.SOLVER.WARMUP_STEPS = 500 #_C.SOLVER.STEPS_PER_EPOCH/2
_C.SOLVER.WARMUP_OVERLAP = True
_C.SOLVER.CHECKPOINT_INTERVAL = 1

########################################################################
# Inference
########################################################################

_C.MODEL.INFERENCE = CN()
_C.MODEL.INFERENCE.USE_BATCHED_NMS = True
_C.MODEL.INFERENCE.POST_NMS_TOPN = 1000
_C.MODEL.INFERENCE.DETECTIONS_PER_IMAGE = 100
_C.MODEL.INFERENCE.DETECTOR_NMS = 0.5
_C.MODEL.INFERENCE.CLASS_AGNOSTIC = False
_C.MODEL.INFERENCE.VISUALIZE_INTERVAL = 250
_C.MODEL.INFERENCE.VISUALIZE_THRESHOLD = 0.75

########################################################################
# Cascade Settings
########################################################################

_C.MODEL.RCNN.CASCADE = CN()
_C.MODEL.RCNN.CASCADE.BBOX_REG_WEIGHTS = [(10., 10., 5., 5.), (20., 20., 10., 10.), (30., 30., 15., 15.)]
_C.MODEL.RCNN.CASCADE.THRESHOLDS = [0.5, 0.6, 0.7]
_C.MODEL.RCNN.CASCADE.STAGE_WEIGHTS = [1.0, 0.5, 0.25]
