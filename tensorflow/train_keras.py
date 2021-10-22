import sys
sys.path.append('..')
import argparse
from configs import cfg
from sagemakercv.detection import build_detector
from sagemakercv.data import build_dataset
from sagemakercv.training import build_optimizer
from sagemakercv.utils.dist_utils import get_dist_info, MPI_size, is_sm_dist
from sagemakercv.data.coco import evaluation
import tensorflow as tf

def is_sm_dist():
    return True

if is_sm_dist():
    import smdistributed.dataparallel.tensorflow.keras as dist
else:
    import horovod.keras as dist

dist.init()

rank, local_rank, size, local_size = get_dist_info()
devices = tf.config.list_physical_devices('GPU')
for device in devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.config.set_visible_devices([devices[local_rank]], 'GPU')
logical_devices = tf.config.list_logical_devices('GPU')
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": cfg.SOLVER.FP16})
tf.config.optimizer.set_jit(cfg.SOLVER.XLA)

def main(cfg):
    dataset = build_dataset(cfg)
    detector_model = build_detector(cfg)

    optimizer = build_optimizer(cfg, keras=True)

    optimizer = dist.DistributedOptimizer(optimizer)

    detector_model.compile(optimizer=optimizer)

    steps_per_epoch = cfg.SOLVER.NUM_IMAGES // cfg.INPUT.TRAIN_BATCH_SIZE
    #epochs = cfg.SOLVER.MAX_ITERS // steps_per_epoch + 1
    #steps_per_epoch = 100
    epochs = 1

    callbacks = [dist.callbacks.BroadcastGlobalVariablesCallback(0)]

    detector_model.fit(x=dataset,
                       steps_per_epoch=steps_per_epoch,
                       epochs=epochs,
                       callbacks=callbacks,
                       verbose=1 if rank == 0 else 0)

def evaluate(cfg, detector_model):
    eval_dataset = build_dataset(cfg, mode='eval')
    coco_prediction = detector_model.predict(x=eval_dataset)
    imgIds, box_predictions, mask_predictions = evaluation.process_prediction(coco_prediction)

    if is_sm_dist():
        from smdistributed.dataparallel.tensorflow import get_worker_comm
        comm = get_worker_comm()
    else:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    box_predictions_mpi_list = comm.gather(box_predictions, root=0)
    mask_predictions_mpi_list = comm.gather(mask_predictions, root=0)

    if rank == 0:
        box_predictions = []
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
