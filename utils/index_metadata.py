"""Copy a FAISS index together with the paths of images injected into it."""

import copy


def clone_image_database(index):
    # FAISS pickling only preserves its native index, not Python attributes.
    cloned = copy.deepcopy(index)
    cloned._aqua_image_paths = dict(getattr(index, "_aqua_image_paths", {}))
    return cloned
