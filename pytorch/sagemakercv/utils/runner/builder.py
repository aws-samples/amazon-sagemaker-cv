from ..registry import Registry

HOOKS = Registry()

def build_hooks(cfg):
    return [HOOKS[hook](cfg) for hook in cfg.HOOKS]
