from pathlib import Path
from hamer.models.hamer import HAMER
from detectron2.config import LazyConfig
import hamer

import pytorch_lightning as pl
import torch
from yacs.config import CfgNode


from typing import Dict

from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from vitpose_model import ViTPoseModel


class SHAMER(HAMER):

    def __init__(
        self,
        cfg: CfgNode,
        init_renderer: bool = False,
    ):

        cfg_path = Path(hamer.__file__).parent/'configs' / \
            'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl" # type: ignore
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25 # type: ignore
        self.detector = DefaultPredictor_Lazy(detectron2_cfg)
        self.cpm = ViTPoseModel(device=torch.device('cuda'))

        super(SHAMER, self).__init__(cfg, init_renderer)
        # TODO: What are the dimensions of the camera params?
        self.camera_parameter_projector = torch.nn.Linear(10, 10)
        # TODO: What are the dimensions of the MANO params?
        self.mano_parameter_projector = torch.nn.Linear(10, 10)

    def configure_optimizers(self):  # type: ignore
        param_groups = [{"params": filter(
            lambda p: p.requires_grad, self.parameters()), "lr": self.cfg.TRAIN.LR}]
        optimizer = torch.optim.AdamW(
            params=param_groups,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
        )
        return optimizer

    def forward_step(
        self,
        batch: Dict,
        train: bool = False
    ) -> Dict:  # type: ignore

        # TODO: Detect human hands, crop accordingly using the detector

        x = batch[0]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W) / 255.0

        conditioning_feats = self.backbone(x[:, :, :, 32:-32])
        pred_mano_params, pred_cam, _ = self.mano_head(conditioning_feats)

        # TODO: What do we do here, design prediction head.
        print(pred_mano_params)

    def compute_loss(
        self,
        batch: Dict,
        output: Dict,
        train: bool = True
    ) -> torch.Tensor:
        raise NotImplementedError(
            "SignHAMER does not implement compute_loss, "
            "as it is not used for training in the current setup."
        )
        # TODO: Apply class specific constrastive loss here

    @pl.utilities.rank_zero.rank_zero_only  # type: ignore
    def tensorboard_logging(self, batch, output, step_count, train=True, write_to_summary_writer=True):
        raise NotImplementedError(
            "SignHAMER does not implement tensorboard_logging, "
            "as it is not used for training in the current setup."
        )
        # TODO: Override this method to log relevant information to TensorBoard
        return super().tensorboard_logging(batch, output, step_count, train, write_to_summary_writer)

    def forward(self, batch: Dict) -> Dict:
        return self.forward_step(batch, train=False)

    def training_step(
        self,
        joint_batch: Dict,
        batch_idx: int
    ) -> Dict:
        batch = joint_batch["clip"]
        optimizer = self.optimizers(use_pl_optimizer=True)
        output = self.forward_step(batch, train=True)
        loss = self.compute_loss(batch, output, train=True)
        optimizer.zero_grad()  # type: ignore
        self.manual_backward(loss)

        if self.cfg.TRAIN.get("GRAD_CLIP_VAL", 0) > 0:
            gn = torch.nn.utils.clip_grad_norm_(
                self.get_parameters(),
                self.cfg.TRAIN.GRAD_CLIP_VAL,
                error_if_nonfinite=True
            )

        optimizer.step()  # type: ignore

        # if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
        #     self.tensorboard_logging(
        #         joint_batch,
        #         output,
        #         self.global_step,
        #         train=True
        #     )

        # TODO: What to actually log here?
        # self.log('train/loss', output['losses']['loss'],
        #          on_step=True, on_epoch=True, prog_bar=True, logger=False)
        return output

    def validation_step(self, batch, batch_idx, dataloader_idx=0) -> Dict:
        return super().validation_step(batch, batch_idx, dataloader_idx)
