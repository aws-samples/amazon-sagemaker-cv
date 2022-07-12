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