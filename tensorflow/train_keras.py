import sys
sys.path.append('..')
import argparse
from configs import cfg
from sagemakercv.detection import build_detector
from sagemakercv.data import build_dataset
from sagemakercv.utils.dist_utils import get_dist_info, MPI_size, is_sm_dist
import tensorflow as tf

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
    detector = build_detector(cfg)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01 * cfg.INPUT.TRAIN_BATCH_SIZE / 8)
    optimizer = dist.DistributedOptimizer(optimizer)
    detector.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=optimizer)

    steps_per_epoch = cfg.SOLVER.NUM_IMAGES // cfg.INPUT.TRAIN_BATCH_SIZE
    epochs = cfg.SOLVER.MAX_ITERS // steps_per_epoch + 1

    callbacks = [dist.callbacks.BroadcastGlobalVariablesCallback(0)]

    detector.fit(x=dataset,
                 steps_per_epoch=steps_per_epoch,
                 epochs=epochs,
                 callbacks=callbacks,
                 verbose=1 if rank == 0 else 0)

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
