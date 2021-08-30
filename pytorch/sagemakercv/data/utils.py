import torch

def prefetcher(load_iterator, device):
    prefetch_stream = torch.cuda.Stream()
    pad_batches = []

    def _prefetch():
        try:
            images, targets, _ = next(load_iterator)
        except StopIteration:
            return None, None

        with torch.cuda.stream(prefetch_stream):
            # TODO:  I'm not sure if the dataloader knows how to pin the targets' datatype.
            targets = [target.to(device, non_blocking=True) for target in targets]
            images = images.to(device, non_blocking=True)

        return images, targets

    next_images, next_targets = _prefetch()

    while next_images is not None:
        torch.cuda.current_stream().wait_stream(prefetch_stream)
        current_images, current_targets = next_images, next_targets
        next_images, next_targets = _prefetch()
        yield current_images, current_targets