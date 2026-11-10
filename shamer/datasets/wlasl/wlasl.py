from __future__ import annotations
import glob
import json
import os
import pathlib
import warnings
import numpy as np
import torchcodec

from yacs.config import CfgNode as CN

from shamer.datasets.clip_dataset import ClipDataset


class WLASLDataset(ClipDataset):

    def __init__(
        self: WLASLDataset,
        cfg: CN,
    ) -> None:

        data_root = pathlib.Path(cfg.DATA_ROOT)

        specification_path = data_root / \
            "metadata" / f"WLASL_{cfg.VERSION}.json"

        with open(specification_path, "r") as f:
            specification = json.load(f)

        # Sort the glosses by their number of occurcences
        specification = sorted(
            specification,
            key=lambda gloss: len(gloss["instances"]),
            reverse=True
        )

        # Filter the specification by the numbered subset
        specification = specification[:cfg.SUBSET]

        # Take only the instances of each gloss that match the mode
        specification = [
            {
                "gloss": gloss["gloss"],
                "instances": list(filter(
                    lambda instance: instance["split"] == cfg.MODE,
                    gloss["instances"]
                ))
            } for gloss in specification
        ]

        videos = [{"gloss": gloss["gloss"], **instance}
                  for gloss in specification for instance in gloss["instances"]]

        samples = []
        for video in videos:

            video_path = data_root / "data" / f"{video['video_id']}.mp4"
            video_decoder = torchcodec.decoders.VideoDecoder(video_path)

            if video_decoder.metadata.num_frames is None:

                warnings.warn(
                    f"Could not read number of frames from metadata for video at \"{video_path}\"")
                continue

            else:

                clip_indices = _calculate_clip_indices(
                    num_frames=video_decoder.metadata.num_frames,
                    frames_per_clip=cfg.FRAMES_PER_CLIP,
                    frame_step=cfg.FRAME_STEP,
                    stride=cfg.STRIDE,
                    pad_start=cfg.PAD_START,
                    pad_end=cfg.PAD_END
                )

                samples.extend([{**video, "path": video_path, "indices": indices}
                               for indices in clip_indices])

        self._glosses = sorted(set(sample["gloss"] for sample in samples))
        samples = [
            {**sample, "class": self._glosses.index(sample["gloss"])} for sample in samples]
        super(WLASLDataset, self).__init__(samples=samples)

    @property
    def glosses(self: WLASLDataset) -> list[str]:
        """Returns the list of glosses in the dataset."""
        return self._glosses


def _calculate_clip_indices(
    num_frames: int,
    frames_per_clip: int,
    frame_step: int,
    stride: int,
    pad_start: bool = False,
    pad_end: bool = False,
) -> list[list[int]]:

    start_indices = np.arange(0, num_frames, step=stride)

    all_indices = []

    for start in start_indices:

        end = min(int(start + frames_per_clip * frame_step), num_frames)
        clip_indices = np.arange(start, end, step=frame_step)

        if pad_start:

            clip_indices = np.pad(
                clip_indices,
                pad_width=(frames_per_clip - len(clip_indices), 0),
                mode="constant",
                constant_values=clip_indices[0]
            )

        if pad_end:

            clip_indices = np.pad(
                clip_indices,
                pad_width=(0, frames_per_clip - len(clip_indices)),
                mode="constant",
                constant_values=clip_indices[-1]
            )

        if len(clip_indices) < frames_per_clip:
            continue

        clip_indices = clip_indices.tolist()
        all_indices.append(clip_indices)

    return all_indices


def prepare_wlasl(
    cfg: CN,
) -> CN:
    import logging
    import tqdm
    import torch

    # Configuring logging

    logging.basicConfig(level=logging.INFO)

    logger = logging.getLogger(__name__)
    logger.info("Preparing WLASL dataset...")

    # Determine run device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determining all paths

    videos_dir = os.path.join(cfg.DATA_ROOT, "data")
    video_paths = glob.glob(os.path.join(videos_dir, "*.mp4"))

    logging.info(f"Found {len(video_paths)} videos in {videos_dir}")

    # Loading the body detector as was given by the demo
    logging.info("Loading body detector...")
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

    # We always use ViTDet
    from detectron2.config import LazyConfig
    import hamer

    cfg_path = pathlib.Path(hamer.__file__).parent / \
        'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"  # type: ignore
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25 # type: ignore
    detector = DefaultPredictor_Lazy(detectron2_cfg)

    # We also need ViTPose to detect the (hand) keypoints
    logging.info("Loading ViTPose model for keypoint detection...")
    from vitpose_model import ViTPoseModel

    cpm = ViTPoseModel(device=device)

    # We ignore the renderer from the demo, since we don't really need it for dataset preparation.

    logging.info("Starting video processing...")
    # We keep track of all videos, that could not be processed
    failed_videos = []

    # Processing each video
    for video_path in tqdm.tqdm(video_paths, desc="Processing videos"):

        # Determine the video ID from the file name, it's a 5 digit number with leading zeros
        video_id = os.path.splitext(os.path.basename(video_path))[0]

        video_decoder = torchcodec.decoders.VideoDecoder(video_path)

        if video_decoder.metadata.num_frames is None:
            warnings.warn(
                f"Could not read number of frames from metadata for video at \"{video_path}\"")
            failed_videos.append(video_path)
            continue

        for frame_idx in range(video_decoder.metadata.num_frames):

            # Decode the frame
            frame = video_decoder.get_frame_at(frame_idx).data
            frame = frame.permute(1, 2, 0).numpy()  # Convert to HWC format

            det

    return CN()
