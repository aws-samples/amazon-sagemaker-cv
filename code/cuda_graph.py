import torch
from function import graph

class GraphedWrapper(torch.nn.Module):
    def __init__(self, model_segment, expected_batch_size_train, graphed_forwards_train, expected_batch_size_test, graphed_forwards_test):
        super().__init__()
        self.model_segment = model_segment
        self.expected_batch_size_train = expected_batch_size_train
        self.graphed_forwards_train = graphed_forwards_train
        self.expected_batch_size_test = expected_batch_size_test
        self.graphed_forwards_test = graphed_forwards_test
    
    def pad_incomplete_batch(self, shape, expected_batch_size, tensor, sizes_tensor, graphed_forwards):
        if shape in graphed_forwards:
            return graphed_forwards[shape](tensor, sizes_tensor)
        elif tensor.shape[0] < expected_batch_size:
            # pad
            before_pad = tensor.shape[0]
            tensor = torch.nn.functional.pad(tensor, (0,0,0,0,0,0,0,expected_batch_size-before_pad))
            sizes_tensor = torch.nn.functional.pad(sizes_tensor, (0,0,0,expected_batch_size-before_pad))
            # run with graph
            shape = tuple(list(tensor.shape))
            if shape in graphed_forwards:
                out = graphed_forwards[shape](tensor, sizes_tensor)
            else:
                out = self.model_segment.eager_forward(tensor, sizes_tensor)
            # unpad
            out = [o[0:before_pad] for o in out]
            return out
        else:
            return self.model_segment.eager_forward(tensor, sizes_tensor)
    
    def forward(self, images_tensor, image_sizes_tensor):
        shape = tuple(list(images_tensor.shape))
        if self.training:
            return self.pad_incomplete_batch(shape, self.expected_batch_size_train, images_tensor, image_sizes_tensor, self.graphed_forwards_train)
        else:
            return self.pad_incomplete_batch(shape, self.expected_batch_size_test, images_tensor, image_sizes_tensor, self.graphed_forwards_test)

def build_graph(model, 
                cfg, 
                images_per_gpu_train, 
                images_per_gpu_test,
                device):
    min_size = cfg.INPUT.MIN_SIZE_TRAIN[0] if isinstance(cfg.INPUT.MIN_SIZE_TRAIN, tuple) else cfg.INPUT.MIN_SIZE_TRAIN
    max_size = cfg.INPUT.MAX_SIZE_TRAIN[0] if isinstance(cfg.INPUT.MAX_SIZE_TRAIN, tuple) else cfg.INPUT.MAX_SIZE_TRAIN
    divisibility = max(1, cfg.DATALOADER.SIZE_DIVISIBILITY)
    shapes_per_orientation = cfg.CUDA_GRAPH_NUM_SHAPES_PER_ORIENTATION
    
    min_size = ((min_size + divisibility - 1) // divisibility) * divisibility
    max_size = ((max_size + divisibility - 1) // divisibility) * divisibility
    size_range = (max_size - min_size) // divisibility
    
    shapes = []
    for i in range(0,shapes_per_orientation):
        size = min_size + ((i+1) * size_range // shapes_per_orientation) * divisibility
        shapes.append( (min_size, size) )
        shapes.append( (size, min_size) )
    print(shapes)
    
    if cfg.MODEL.BACKBONE.DONT_RECOMPUTE_SCALE_AND_BIAS:
        model.compute_scale_bias() # Enable caching of scale and bias for frozen batchnorms
    per_gpu_batch_sizes = [(True, images_per_gpu_train), (False, images_per_gpu_test)]
    print("USE_CUDA_GRAPH :: per_gpu_batch_sizes = %s" % (str(per_gpu_batch_sizes)))
    graphed_forwards_train, graphed_forwards_test = {}, {}
    graph_stream = torch.cuda.Stream()
    for (is_training, images_per_gpu) in per_gpu_batch_sizes:
        if is_training:
            model.train()
        else:
            model.eval()
        for i, shape in enumerate(shapes):
            dummy_shape = (images_per_gpu,) + shape + (3,) if cfg.NHWC else (images_per_gpu,3,) + shape
            dummy_batch = torch.ones(dummy_shape, dtype=torch.float16, device=device)
            dummy_image_sizes = torch.tensor([list(shape) for _ in range(images_per_gpu)], dtype=torch.float32, device=device)
            sample_args = (dummy_batch.clone(),dummy_image_sizes.clone(),)
            forward_fn = "graph_forward_%s_%d_%d" % ("train" if is_training else "test", images_per_gpu, i+1)
            if i == 0:
                model.graphable = graph(model.graphable,
                                           sample_args,
                                           graph_stream=graph_stream,
                                           warmup_only=True,
                                           overwrite_fn='eager_forward')
                model.graphable, pool_id = graph(model.graphable,
                                                    sample_args,
                                                    graph_stream=graph_stream,
                                                    warmup_only=False,
                                                    overwrite_fn=forward_fn,
                                                    return_pool_id=True)
            else:
                model.graphable = graph(model.graphable,
                                           sample_args,
                                           graph_stream=graph_stream,
                                           warmup_only=False,
                                           overwrite_fn=forward_fn,
                                           use_pool_id=pool_id)
            if is_training:
                graphed_forwards_train[dummy_shape] = getattr(model.graphable, forward_fn)
            else:
                graphed_forwards_test[dummy_shape] = getattr(model.graphable, forward_fn)
            
        model.graphable = GraphedWrapper(model.graphable, images_per_gpu_train, graphed_forwards_train, images_per_gpu_test, graphed_forwards_test)
        return model, shapes