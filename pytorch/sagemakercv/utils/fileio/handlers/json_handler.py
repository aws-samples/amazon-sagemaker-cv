# Copyright (c) Open-MMLab. All rights reserved.
import json
import numpy as np

from .base import BaseFileHandler

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

class JsonHandler(BaseFileHandler):

    def load_from_fileobj(self, file):
        return json.load(file)

    def dump_to_fileobj(self, obj, file, **kwargs):
        obj = json.dumps(obj, cls=NumpyEncoder)
        json.dump(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        return json.dumps(obj, **kwargs)
