import torch

from maskrcnn_benchmark.utils.comm import synchronize, get_rank, is_main_process, get_world_size, is_main_evaluation_process
from maskrcnn_benchmark.utils.async_evaluator import init, get_evaluator, set_epoch_tag, get_tag
from maskrcnn_benchmark.utils.mlperf_logger import log_end, log_start, log_event, generate_seeds, broadcast_seeds, barrier, configure_logger
from maskrcnn_benchmark.utils.async_evaluator import init, get_evaluator, set_epoch_tag, get_tag
from maskrcnn_benchmark.utils.timed_section import TimedSection
import logging

def mlperf_test_early_exit(iteration, iters_per_epoch, tester, model, distributed, min_bbox_map, min_segm_map, world_size):
    # Note: let iters / epoch == 10k, at iter 9999 we've finished epoch 0 and need to test
    if iteration > 0 and (iteration + 1)% iters_per_epoch == 0:
        synchronize(comm=None)
        epoch = iteration // iters_per_epoch + 1

        # log_end(key=constants.EPOCH_STOP, metadata={"epoch_num": epoch})
        # log_end(key=constants.BLOCK_STOP, metadata={"first_epoch_num": epoch})
        # sbridge.start_eval_prof()
        # log_start(key=constants.EVAL_START, metadata={"epoch_num":epoch})
        # set the async evaluator's tag correctly
        set_epoch_tag(epoch)

        # Note: No longer returns anything, underlying future is in another castle
        tester(model=model, distributed=distributed)
        # necessary for correctness
        model.train()
    elif iteration % 10 == 9: # do finished check after every 10 iterations
        # Otherwise, check for finished async results
        results = check_completed_tags(iteration, world_size)

        # on master process, check each result for terminating condition
        # sentinel for run finishing
        finished = 0
        if is_main_process():
            for result_epoch, (bbox_map, segm_map) in results.items():
                logger = logging.getLogger('maskrcnn_benchmark.trainer')
                logger.info('bbox mAP: {}, segm mAP: {}'.format(bbox_map, segm_map))

                # log_event(key=constants.EVAL_ACCURACY, value={"BBOX" : bbox_map, "SEGM" : segm_map}, metadata={"epoch_num" : result_epoch} )
                # sbridge.stop_eval_prof()
                # log_end(key=constants.EVAL_STOP, metadata={"epoch_num": result_epoch})
                # terminating condition
                if bbox_map >= min_bbox_map and segm_map >= min_segm_map:
                    logger.info("Target mAP reached, exiting...")
                    finished = 1
                    #return True

        # We now know on rank 0 whether or not we should terminate
        # Bcast this flag on multi-GPU
        if world_size > 1:
            with torch.no_grad():
                finish_tensor = torch.tensor([finished], dtype=torch.int32, device = torch.device('cuda'))
                torch.distributed.broadcast(finish_tensor, 0)

                # If notified, end.
                if finish_tensor.item() == 1:
                    return True #, sbridge
        else:
            # Single GPU, don't need to create tensor to bcast, just use value directly
            if finished == 1:
                return True #, sbridge

    # Otherwise, default case, continue
    return False #, sbridge

finished_prep_work = None

def check_completed_tags(iteration, world_size, dedicated_evalution_ranks=0, eval_ranks_comm=None):
    # Check for completeness is fairly expensive, so we only do it once per N iterations
    # Only applies when not using dedicated evaluation ranks
    if dedicated_evalution_ranks == 0 and iteration % 10 != 9:
        return {}

    num_evaluation_ranks = world_size if dedicated_evalution_ranks == 0 else dedicated_evalution_ranks

    global finished_prep_work
    from maskrcnn_benchmark.data.datasets.evaluation.coco.coco_eval import COCOResults, all_gather_prep_work, evaluate_coco
    if num_evaluation_ranks > 1:
        num_finished = torch.zeros([1], dtype=torch.int32, device='cuda') if finished_prep_work is None else torch.ones([1], dtype=torch.int32, device='cuda')
        torch.distributed.all_reduce(num_finished, group=eval_ranks_comm)
        ready_to_submit_evaluation_task = True if num_finished == num_evaluation_ranks else False
    else:
        ready_to_submit_evaluation_task = False if finished_prep_work is None else True
    evaluator = get_evaluator()
    if ready_to_submit_evaluation_task:
        with TimedSection("EXPOSED: Launching evaluation task took %.3fs"):
            coco_results, iou_types, coco, output_folder = finished_prep_work
            finished_prep_work = None
            coco_results = all_gather_prep_work(coco_results, dedicated_evalution_ranks, eval_ranks_comm)
            if is_main_evaluation_process(dedicated_evalution_ranks):
                evaluator.submit_task(get_tag(),
                                      evaluate_coco,
                                      coco,
                                      coco_results,
                                      iou_types,
                                      output_folder)
    else:
        # loop over all all epoch, result pairs that have finished
        all_results = {}
        for t, r in evaluator.finished_tasks().items():
            # Note: one indirection due to possibility of multiple test datasets
            # we only care about the first
            map_results = r# [0]
            if isinstance(map_results, COCOResults):
                bbox_map = map_results.results["bbox"]['AP']
                segm_map = map_results.results["segm"]['AP']
                all_results.update({ t : (bbox_map, segm_map) })
            else:
                finished_prep_work = map_results

        return all_results

    return {}