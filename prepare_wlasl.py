import argparse

from shamer.datasets.wlasl.config import get_defaults
from shamer.datasets.wlasl.wlasl import prepare_wlasl

parser = argparse.ArgumentParser(
    description="Prepare WLASL dataset to use with SignHAMER. Hands are detected and cropped from all videos.")
parser.add_argument("--data-root", type=str, required=False)

args = parser.parse_args()

cfg = get_defaults()
cfg.defrost()
cfg.DATA_ROOT = args.data_root if args.data_root else cfg.DATA_ROOT

prepare_wlasl(
    cfg=cfg
)
