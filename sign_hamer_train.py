from yacs.config import CfgNode
from hamer.models import DEFAULT_CHECKPOINT, download_models
from hamer.configs import CACHE_DIR_HAMER, get_config
import torch
from typing import Optional
from pathlib import Path
import argparse
import pyrootutils
import pytorch_lightning as pl

from shamer.datasets import SHAMERDataModule
from shamer.datasets.wlasl.config import get_defaults
from shamer.models.sign_hamer import SHAMER

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


def load_sign_hamer(checkpoint_path=DEFAULT_CHECKPOINT) -> tuple[SHAMER, CfgNode]:
    model_cfg = str(Path(checkpoint_path).parent.parent / 'model_config.yaml')
    model_cfg = get_config(model_cfg, update_cachedir=True)

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        assert model_cfg.MODEL.IMAGE_SIZE == 256, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()

    # Update config to be compatible with demo
    if ('PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE):
        model_cfg.defrost()
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
        model_cfg.freeze()

    model = SHAMER.load_from_checkpoint(
        checkpoint_path,
        strict=False,
        cfg=model_cfg
    )
    return model, model_cfg


def train():

    parser = argparse.ArgumentParser(description='HaMeR training script')
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to pretrained model checkpoint")

    args = parser.parse_args()

    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_sign_hamer(args.checkpoint)

    data_module = SHAMERDataModule(
        cfg=model_cfg,
        dataset_cfg=get_defaults()
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )

    trainer.fit(model=model, datamodule=data_module)
    print("Training completed successfully!")


def main() -> Optional[float]:
    train()


if __name__ == "__main__":
    main()
