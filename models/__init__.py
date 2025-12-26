from .segmentation import build


def build_model(args, losses):
    return build(args, losses)
