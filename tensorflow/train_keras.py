import sys
import os
sys.path.append('..')
import argparse
from configs import cfg
from sagemakercv.detection import build_detector
from sagemakercv.data import build_dataset
from sagemakercv.training import build_optimizer
from sagemakercv.utils.dist_utils import get_dist_info, MPI_size
from sagemakercv.data.coco import evaluation
import tensorflow as tf

import smdistributed.dataparallel.tensorflow.keras as dist

dist.init()

rank, local_rank, size, local_size = get_dist_info()
devices = tf.config.list_physical_devices('GPU')
for device in devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.config.set_visible_devices([devices[local_rank]], 'GPU')
logical_devices = tf.config.list_logical_devices('GPU')
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": cfg.SOLVER.FP16})
tf.config.optimizer.set_jit(cfg.SOLVER.XLA)

instance_type = os.getenv("SAGEMAKER_INSTANCE_TYPE")
if instance_type != "ml.p4d.24xlarge":
    print('Warning: instance type is not fully supported in private preview, please use p4d.24xlarge for best performance')

# load backbone weights
def load_pretrained_weights(detector_model, dataset, cfg):
    print('Loading checkpoint')
    # populate weights for backbone through a forward pass
    features, labels = next(iter(dataset))
    _ = detector_model(features, training=False)

    chkp = tf.compat.v1.train.NewCheckpointReader(cfg.PATHS.WEIGHTS)
    weights = [chkp.get_tensor(i) for i in ['/'.join(i.name.split('/')[-2:]).split(':')[0] \
                                                for i in detector_model.layers[0].weights]]
    detector_model.layers[0].set_weights(weights)

# main training entry point
def main(cfg):
    dataset = build_dataset(cfg)
    detector_model = build_detector(cfg)

    load_pretrained_weights(detector_model, dataset, cfg)

    optimizer = build_optimizer(cfg, keras=True)

    # SMDDP optimized allreduce is wrapped into Distributed Optimizer
    optimizer = dist.DistributedOptimizer(optimizer)

    detector_model.compile(optimizer=optimizer)

    steps_per_epoch = cfg.SOLVER.NUM_IMAGES // cfg.INPUT.TRAIN_BATCH_SIZE
    epochs = cfg.SOLVER.MAX_ITERS // steps_per_epoch + 1

    callbacks = [dist.callbacks.BroadcastGlobalVariablesCallback(0)]

    # TwoStageDetector model
    detector_model.fit(x=dataset,
                       steps_per_epoch=steps_per_epoch,
                       epochs=epochs,
                       callbacks=callbacks,
                       verbose=1 if rank == 0 else 0)

# distributed evaluation
def evaluate(cfg, detector_model):
    eval_dataset = build_dataset(cfg, mode='eval')
    coco_prediction = detector_model.predict(x=eval_dataset)
    imgIds, box_predictions, mask_predictions = evaluation.process_prediction(coco_prediction)

    from smdistributed.dataparallel.tensorflow import get_worker_comm
    comm = get_worker_comm()

    imgIds_mpi_list = comm.gather(imgIds, root=0)
    box_predictions_mpi_list = comm.gather(box_predictions, root=0)
    mask_predictions_mpi_list = comm.gather(mask_predictions, root=0)

    if rank == 0:
        imgIds = []
        box_predictions = []
        mask_predictions = []

        for i in imgIds_mpi_list:
            imgIds.extend(i)
        print("Running Evaluation for {} images".format(len(set(imgIds))))

        for i in box_predictions_mpi_list:
            box_predictions.extend(i)
        predictions = {'bbox': box_predictions}

        if cfg.MODEL.INCLUDE_MASK:
            for i in mask_predictions_mpi_list:
                mask_predictions.extend(i)
            predictions['segm'] = mask_predictions

        stat_dict = evaluation.evaluate_coco_predictions(cfg.PATHS.VAL_ANNOTATIONS, predictions.keys(), predictions, verbose=False)
        print(stat_dict)

def parse():
    parser = argparse.ArgumentParser(description='Load model configuration')
    parser.add_argument('--config', help='Configuration file to apply on top of base')
    parsed, _ = parser.parse_known_args()
    return parsed

if __name__=='__main__':
    args = parse()
    cfg.merge_from_file(args.config)
    assert cfg.INPUT.TRAIN_BATCH_SIZE%MPI_size()==0, f"Batch {cfg.INPUT.TRAIN_BATCH_SIZE} on {MPI_size()} GPUs"
    assert cfg.INPUT.EVAL_BATCH_SIZE%MPI_size()==0, f"Batch {cfg.INPUT.EVAL_BATCH_SIZE} on {MPI_size()} GPUs"
    main(cfg)
